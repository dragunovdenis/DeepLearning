//Copyright (c) 2022 Denys Dragunov, dragunovdenis@gmail.com
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
//copies of the Software, and to permit persons to whom the Software is furnished
//to do so, subject to the following conditions :

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
//PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
//HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
//OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "CppUnitTest.h"
#include <Math/Vector.h>
#include <Math/ActivationFunction.h>
#include <Math/CostFunction.h>
#include <Utilities.h>
#include <numeric>
#include "StandardTestUtils.h"
#include "ActivationFuncTestUtils.h"
#include "Math/Functions.h"
#include "Math/Tensor.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(ActivationFunctionTest)
	{
	public:
		TEST_METHOD(SigmoidFunctionTest)
		{
			//Arrange
			constexpr auto dim = 10;
			const auto vector = Vector(dim, -1, 1);

			//Act
			const auto result = ActivationFunction<Vector>(ActivationFunctionId::SIGMOID)(vector);

			//Assert
			Assert::AreEqual(vector.dim(), result.dim(), L"Unexpected dimension of the result vector");

			for (std::size_t item_id = 0; item_id < vector.dim(); item_id++)
			{
				const auto diff = std::abs(result[item_id] - Func::sigmoid(vector[item_id]));
				StandardTestUtils::LogAndAssertLessOrEqualTo("Difference",
					diff, std::numeric_limits<Real>::epsilon());
			}
		}

		TEST_METHOD(SigmoidFunctionAndDerivativeTest)
		{
			//Arrange
			constexpr auto dim = 10;
			const auto vector = Vector(dim, -1, 1);

			//Act
			const auto [result, result_deriv] = ActivationFunction<Vector>(ActivationFunctionId::SIGMOID).func_and_aux(vector);

			//Assert
			Assert::AreEqual(vector.dim(), result.dim(), L"Unexpected dimension of the result vector");
			Assert::AreEqual(vector.dim(), result_deriv.dim(), L"Unexpected dimension of the result derivative vector");

			//Here we use the activation function to generate the reference values, because "()" operator of the activation function
			//is tested separately, and we rely on that
			const auto result_reference = ActivationFunction<Vector>(ActivationFunctionId::SIGMOID)(vector);
			Assert::IsTrue((result_reference - result).max_abs() <= 0, L"Unexpectedly high deviation from the function reference value.");

			for (std::size_t item_id = 0; item_id < vector.dim(); item_id++)
			{
				const auto deriv_diff = std::abs(result_deriv[item_id] - Func::sigmoid_deriv(vector[item_id]));
				StandardTestUtils::LogAndAssertLessOrEqualTo("Derivative difference",
					deriv_diff, std::numeric_limits<Real>::epsilon());
			}
		}

		/// <summary>
		/// "Reference" implementation of "soft-max' function
		/// </summary>
		Vector soft_max_reference(const Vector& vec) const
		{
			Vector exponents(vec.size());

			std::ranges::transform(vec, exponents.begin(), [](const auto& x) { return std::exp(x); });
			const auto sum_of_exponents = std::accumulate(exponents.begin(), exponents.end(), static_cast<Real>(0));
			return exponents * (static_cast<Real>(1) / sum_of_exponents);
		}

		TEST_METHOD(SoftMaxFunctionTest)
		{
			//Arrange
			constexpr auto dim = 10;
			const SoftMaxActivationFunction<Vector> soft_max_activation_func;
			const auto vec = Vector(dim, -1, 1);
			Assert::IsTrue(vec.max_abs() > 0, L"Vector is supposed to be nonzero");

			//Act
			const auto soft_max_result = soft_max_activation_func(vec);

			//Assert
			const auto soft_max_reference_result = soft_max_reference(vec);
			const auto diff = (soft_max_result - soft_max_reference_result).max_abs();
			StandardTestUtils::LogAndAssertLessOrEqualTo("Difference",
				diff, std::numeric_limits<Real>::epsilon());
		}

		TEST_METHOD(SoftMaxFunctionAndDerivativeTest)
		{
			//Arrange
			constexpr auto dim = 10;
			const SoftMaxActivationFunction<Vector> soft_max_activation_func;
			const auto quadratic_cost_func = CostFunction<Vector>(CostFunctionId::SQUARED_ERROR);
			const auto input = Vector(dim, -1, 1);
			const auto reference = Vector(dim, -1, 1);
			Assert::IsTrue(input.max_abs() > 0, L"Vector is supposed to be nonzero");

			//Act
			const auto [soft_max_result, aux_data] = soft_max_activation_func.func_and_aux(input);
			const auto [cost, gradient] = quadratic_cost_func.func_and_deriv(soft_max_result, reference);
			const auto activation_input_gradient = soft_max_activation_func.get_in_grad(gradient, aux_data);

			//Assert
			const auto soft_max_reference_result = soft_max_reference(input);
			const auto diff = (soft_max_result - soft_max_reference_result).max_abs();
			StandardTestUtils::LogAndAssertLessOrEqualTo("Difference",
				diff, std::numeric_limits<Real>::epsilon());

			constexpr auto double_precision = std::is_same_v<Real, double>;
			constexpr auto delta = double_precision ? 1e-5 : static_cast<Real>(1e-2);
			constexpr auto diff_threshold = double_precision ? 7e-11 : static_cast<Real>(3e-5);
			for (auto element_id = 0ull; element_id < input.size(); element_id++)
			{
				auto input_plus_delta = input;
				input_plus_delta[element_id] += delta;
				const auto result_plus_delta = quadratic_cost_func(soft_max_reference(input_plus_delta), reference);

				auto input_minus_delta = input;
				input_minus_delta[element_id] -= delta;
				const auto result_minus_delta = quadratic_cost_func(soft_max_reference(input_minus_delta), reference);

				const auto deriv_numeric = (result_plus_delta - result_minus_delta) / (2 * delta);

				const auto deriv_diff = std::abs(deriv_numeric - activation_input_gradient[element_id]);

				StandardTestUtils::LogAndAssertLessOrEqualTo("Derivative difference",
					deriv_diff, diff_threshold);
			}
		}

		/// <summary>
		/// General method to test "add-in-gradient" functionality of different cost functions.
		/// </summary>
		template <template <class> class F>
		static void function_add_in_grad_test()
		{
			constexpr auto dim = 10;
			const F<Vector> func{};
			const auto out_grad = Vector(dim, -1, 1);
			const auto aux_data = Vector(dim, -1, 1);
			const auto init_value = Vector(dim, -1, 1);
			Assert::IsTrue(out_grad.max_abs() > 0 && aux_data.max_abs() &&
				init_value.max_abs() > 0, L"Vector is supposed to be nonzero");

			//Act
			auto result = init_value;
			func.add_in_grad(out_grad, aux_data, result);

			// Assert
			const auto result_expected = func.get_in_grad(out_grad, aux_data) + init_value;
			const auto diff = (result - result_expected).max_abs();
			StandardTestUtils::LogAndAssertLessOrEqualTo("Difference",
				diff, std::numeric_limits<Real>::epsilon());
		}

		TEST_METHOD(SoftMaxFunctionAddInGradTest)
		{
			function_add_in_grad_test<SoftMaxActivationFunction>();
		}

		TEST_METHOD(SigmoidFunctionAddInGradTest)
		{
			function_add_in_grad_test<SigmoidActivationFunction>();
		}

		TEST_METHOD(ReLuOptimizedVsGeneralTest)
		{
			RunOptimizedVsGeneralActivationFuncTest(ReLuActivationFunction<Tensor>(), ActivationFunctionId::RELU);
		}

		TEST_METHOD(SigmoidOptimizedVsGeneralTest)
		{
			RunOptimizedVsGeneralActivationFuncTest(SigmoidActivationFunction<Tensor>(), ActivationFunctionId::SIGMOID);
		}

		TEST_METHOD(TanhOptimizedVsGeneralTest)
		{
			RunOptimizedVsGeneralActivationFuncTest(TanhActivationFunction<Tensor>(), ActivationFunctionId::TANH);
		}
	};
}
