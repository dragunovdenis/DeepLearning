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
#include <Math/CudaVector.cuh>
#include <Math/ActivationFunction.h>
#include <Math/CostFunction.h>
#include <Utilities.h>
#include <numeric>
#include "StandardTestUtils.h"

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
			const auto result = ActivationFunction<Vector>(ActivationFunctionId::SIGMOID)(vector);

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

		TEST_METHOD(SigmoidFunctionCudaTest)
		{
			//Arrange
			const auto dim = 10000;
			const auto vector = CudaVector(dim, -1, 1);

			//Act
			const auto result = ActivationFunction<CudaVector>(ActivationFunctionId::SIGMOID)(vector);

			//Assert
			const auto result_host = ActivationFunction<Vector>(ActivationFunctionId::SIGMOID)(vector.to_host());
			const auto diff = (result.to_host() - result_host).max_abs();

			Logger::WriteMessage((std::string("Diff = ") + Utils::to_string(diff) + "\n").c_str());
			Assert::IsTrue(diff < 10 * std::numeric_limits<Real>::epsilon(),
				L"Unexpectedly high deviation from the reference value.");
		}

		TEST_METHOD(SigmoidFunctionAndDerivativeTest)
		{
			//Arrange
			const auto dim = 10;
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
				const auto deriv_diff = std::abs(result_deriv(item_id) - sigmoid_deriv(vector(item_id)));
				Logger::WriteMessage((std::string("Derivative difference = ") + std::to_string(deriv_diff) + "\n").c_str());
				Assert::IsTrue(deriv_diff < std::numeric_limits<Real>::epsilon(),
					L"Unexpectedly high deviation from the derivative reference value.");
			}
		}

		TEST_METHOD(SigmoidFunctionAndDerivativeCudaTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector = CudaVector(dim, -1, 1);
			const auto out_grad = CudaVector(dim, -1, 1);
			Assert::IsTrue(vector.max_abs() > 0 && out_grad.max_abs() > 0, L"Vectors are supposed to be nonzero");

			//Act
			const auto function = ActivationFunction<CudaVector>(ActivationFunctionId::SIGMOID);
			const auto [result, result_aux] = function.func_and_aux(vector);
			const auto result_gradient = function.calc_input_gradient(out_grad, result_aux);

			//Assert
			const auto function_host = ActivationFunction<Vector>(ActivationFunctionId::SIGMOID);
			const auto [result_host, result_aux_host] = function_host.func_and_aux(vector.to_host());
			const auto result_gradient_host = function_host.calc_input_gradient(out_grad.to_host(), result_aux.to_host());

			Assert::IsTrue((result_host - result.to_host()).max_abs() < 10 * std::numeric_limits<Real>::epsilon(), 
				L"Result: too high deviation from reference");
			Assert::IsTrue((result_aux_host - result_aux.to_host()).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Auxiliary data: Too high deviation from reference");
			Assert::IsTrue((result_gradient_host - result_gradient.to_host()).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Gradient: Too high deviation from reference");
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
			const SoftMaxActivationFunction<Vector> soft_max_activation_func;
			const auto vec = Vector(dim, -1, 1);
			Assert::IsTrue(vec.max_abs() > 0, L"Vector is supposed to be nonzero");

			//Act
			const auto soft_max_result = soft_max_activation_func(vec);

			//Assert
			const auto soft_max_reference_result = soft_max_reference(vec);
			const auto diff = (soft_max_result - soft_max_reference_result).max_abs();
			Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(), L"Too high deviation between the actual and expected values");
		}

		TEST_METHOD(SoftMaxFunctionCudaTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vec = CudaVector(dim, -1, 1);
			Assert::IsTrue(vec.max_abs() > 0, L"Vector is supposed to be nonzero");

			//Act
			const auto soft_max_result = SoftMaxActivationFunction<CudaVector>()(vec);

			//Assert
			const auto soft_max_result_host = SoftMaxActivationFunction<Vector>()(vec.to_host());
			const auto diff = (soft_max_result.to_host() - soft_max_result_host).max_abs();
			Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(), L"Too high deviation between the actual and expected values");
		}

		TEST_METHOD(SoftMaxFunctionAndDerivativeTest)
		{
			//Arrange
			const auto dim = 10;
			const SoftMaxActivationFunction<Vector> soft_max_activation_func;
			const auto quadratic_cost_func = CostFunction<Vector>(CostFunctionId::SQUARED_ERROR);
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

		TEST_METHOD(SoftMaxFunctionAndDerivativeCudaTest)
		{
			//Arrange
			const auto dim = 10;
			const auto input = CudaVector(dim, -1, 1);
			const auto out_grad = CudaVector(dim, -1, 1);
			Assert::IsTrue(input.max_abs() > 0 && out_grad.max_abs() > 0, L"Vectors are supposed to be nonzero");

			//Act
			const auto [soft_max_result, aux_data] = SoftMaxActivationFunction<CudaVector>().func_and_aux(input);
			const auto activation_input_gradient = SoftMaxActivationFunction<CudaVector>().calc_input_gradient(out_grad, aux_data);

			//Assert
			const auto [soft_max_result_host, aux_data_host] = SoftMaxActivationFunction<Vector>().func_and_aux(input.to_host());
			const auto activation_input_gradient_host = SoftMaxActivationFunction<Vector>().calc_input_gradient(out_grad.to_host(), aux_data.to_host());
			const auto diff_func = (soft_max_result.to_host() - soft_max_result_host).max_abs();
			const auto diff_grad = (activation_input_gradient.to_host() - activation_input_gradient_host).max_abs();
			Assert::IsTrue(diff_func < std::numeric_limits<Real>::epsilon(), L"Too high deviation between the actual and expected values (function)");
			Assert::IsTrue(diff_grad < std::numeric_limits<Real>::epsilon(), L"Too high deviation between the actual and expected values (gradient)");
		}
	};
}
