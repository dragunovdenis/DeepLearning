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

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(ActivationFunctionTest)
	{
	public:
		/// <summary>
		/// Sigmoid function
		/// </summary>
		static Real sigmoid(const Real& x)
		{
			return Real(1) / (Real(1) + std::exp(-x));
		}

		/// <summary>
		/// Derivative of the sigmoid function
		/// </summary>
		static Real sigmoid_deriv(const Real& x)
		{
			const auto exp_x = std::exp(-x);
			const auto denominator = (Real(1) + exp_x);
			return exp_x / (denominator * denominator);
		}

		TEST_METHOD(SigmoidFunctionTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector = Vector(dim, -1, 1);

			//Act
			const auto result = ActivationFuncion<Vector>(ActivationFunctionId::SIGMOID)(vector);

			//Assert
			Assert::AreEqual(vector.dim(), result.dim(), L"Unexpected dimension of the result vector");

			for (std::size_t item_id = 0; item_id < vector.dim(); item_id++)
			{
				const auto diff = std::abs(result(item_id) - sigmoid(vector(item_id)));
				Logger::WriteMessage((std::string("Difference = ") + std::to_string(diff) + "\n").c_str());
				Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(),
					L"Unexpectedly high deviation from the reference value.");
			}
		}

		TEST_METHOD(SigmoidFunctionAndDerivativeTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector = Vector(dim, -1, 1);

			//Act
			const auto [result, result_deriv] = ActivationFuncion<Vector>(ActivationFunctionId::SIGMOID).func_and_aux(vector);

			//Assert
			Assert::AreEqual(vector.dim(), result.dim(), L"Unexpected dimension of the result vector");
			Assert::AreEqual(vector.dim(), result_deriv.dim(), L"Unexpected dimension of the result derivative vector");

			//Here we use the activation function to generate the reference values, because "()" operator of the activation function
			//is tested separately, and we rely on that
			const auto result_reference = ActivationFuncion<Vector>(ActivationFunctionId::SIGMOID)(vector);
			Assert::IsTrue((result_reference - result).max_abs() <= 0, L"Unexpectedly high deviation from the function reference value.");

			for (std::size_t item_id = 0; item_id < vector.dim(); item_id++)
			{
				const auto deriv_diff = std::abs(result_deriv(item_id) - sigmoid_deriv(vector(item_id)));
				Logger::WriteMessage((std::string("Derivative difference = ") + std::to_string(deriv_diff) + "\n").c_str());
				Assert::IsTrue(deriv_diff < std::numeric_limits<Real>::epsilon(),
					L"Unexpectedly high deviation from the derivative reference value.");
			}
		}

		/// <summary>
		/// "Reference" implementation of "soft-max' function
		/// </summary>
		Vector soft_max_reference(const Vector& vec)
		{
			Vector exponents(vec.size());

			std::transform(vec.begin(), vec.end(), exponents.begin(), [](const auto& x) { return std::exp(x); });
			const auto sum_of_exponents = std::accumulate(exponents.begin(), exponents.end(), Real(0));
			return exponents * (Real(1) / sum_of_exponents);
		}

		TEST_METHOD(SoftMaxFunctionTest)
		{
			//Arrange
			const auto dim = 10;
			const SoftMaxActivationFuncion<Vector> soft_max_activation_func;
			const auto vec = Vector(dim, -1, 1);
			Assert::IsTrue(vec.max_abs() > 0, L"Vector is supposed to be nonzero");

			//Act
			const auto soft_max_result = soft_max_activation_func(vec);

			//Assert
			const auto soft_max_reference_result = soft_max_reference(vec);
			const auto diff = (soft_max_result - soft_max_reference_result).max_abs();
			Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(), L"Too high deviation between the actual and expected values");
		}

		TEST_METHOD(SoftMaxFunctionAndDerivativeTest)
		{
			//Arrange
			const auto dim = 10;
			const SoftMaxActivationFuncion<Vector> soft_max_activation_func;
			const auto quadratic_cost_func = CostFunction(CostFunctionId::SQUARED_ERROR);
			const auto input = Vector(dim, -1, 1);
			const auto reference = Vector(dim, -1, 1);
			Assert::IsTrue(input.max_abs() > 0, L"Vector is supposed to be nonzero");

			//Act
			const auto [soft_max_result, aux_data] = soft_max_activation_func.func_and_aux(input);
			const auto [cost, gradient] = quadratic_cost_func.func_and_deriv(soft_max_result, reference);
			const auto activation_input_gradient = soft_max_activation_func.calc_input_gradient(gradient, aux_data);

			//Assert
			const auto soft_max_reference_result = soft_max_reference(input);
			const auto diff = (soft_max_result - soft_max_reference_result).max_abs();
			Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(), L"Too high deviation between the actual and expected values");

			const auto delta = Real(1e-5);
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

				Logger::WriteMessage((std::string("Derivative difference = ") + Utils::to_string(deriv_diff) + "\n").c_str());
				Assert::IsTrue(deriv_diff < 7e-11, L"Too high deviation between the actual and expected values");
			}
		}
	};
}
