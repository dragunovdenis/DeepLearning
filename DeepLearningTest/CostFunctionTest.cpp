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
#include <Math/CostFunction.h>
#include "StandardTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(CostFunctionTest)
	{
		/// <summary>
		/// Squared error reference function
		/// </summary>
		static Real SquaredErrorFunc(const Real x, const Real ref)
		{
			const auto diff = x - ref;
			return Real(0.5) * diff * diff;
		}

		/// <summary>
		/// Derivative of the squared error function
		/// </summary>
		static Real SquaredErrorDerivFunc(const Real x, const Real ref)
		{
			return x - ref;
		}

		/// <summary>
		/// Cross-entropy function
		/// </summary>
		static Real CrossEntropyFunc(const Real x, const Real ref)
		{
			return -(ref * log(x) + (Real(1) - ref) * log(1 - x));
		}

		/// <summary>
		/// Derivative of the cross-entropy function
		/// </summary>
		static Real CrossEntropyDerivFunc(const Real x, const Real ref)
		{
			return (x - ref) / (x * (Real(1) - x));
		}

		/// <summary>
		/// General method to test function only (without derivative) evaluation part of the cost function component
		/// </summary>
		template <class RF>
		static void CheckFunctionOnly(const CostFunctionId& func_id, const RF& reference_func)
		{
			//Arrange
			const auto dim = 10;
			const auto cost_func = CostFunction<Vector>(func_id);
			const auto input = Vector(dim, Real(1e-5), Real(1-1e-5));
			const auto reference = Vector(dim, Real(1e-5), Real(1 - 1e-5));
			Assert::IsTrue(input != reference, L"Reference and input vectors should not be equal.");

			//Act
			const auto result = cost_func(input, reference);

			//Assert
			auto result_expected = Real(0);
			for (std::size_t item_id = 0; item_id < dim; item_id++)
				result_expected += reference_func(input(item_id), reference(item_id));

			const auto diff = std::abs(result - result_expected);
			Assert::IsTrue(diff <= 0, L"Unexpectedly high deviation from the reference value");

		}

		/// <summary>
		/// General method to test function and derivative evaluation part of the cost function component
		/// </summary>
		template <class RF>
		static void CheckFunctionAndDerivative(const CostFunctionId& func_id, const RF& reference_func, const RF& reference_deriv)
		{
			//Arrange
			const auto dim = 10;
			const auto cost_func = CostFunction<Vector>(func_id);
			const auto input = Vector(dim, Real(0.1), Real(0.9));
			const auto reference = Vector(dim, Real(0.1), Real(0.9));
			Assert::IsTrue(input != reference, L"Reference and input vectors should not be equal.");

			//Act
			const auto [result_func, result_deriv] = cost_func.func_and_deriv(input, reference);
			const auto result_deriv_single = cost_func.deriv(input, reference);

			//Assert
			Assert::IsTrue(result_deriv.dim() == input.dim() && result_deriv_single.dim() == input.dim(),
				L"Input and result vectors should be of the same dimension.");

			//We use the cost function here to generate the reference because "()" operator of the
			//cost function is tested separately and here we rely on that
			auto result_func_expected = cost_func(input, reference);

			const auto diff_func = std::abs(result_func - result_func_expected);
			Assert::IsTrue(diff_func <= 0, L"Unexpectedly high deviation from the reference function value.");

			for (std::size_t item_id = 0; item_id < dim; item_id++)
			{
				const auto diff_deriv = std::abs(result_deriv(item_id) - reference_deriv(input(item_id), reference(item_id)));
				const auto diff_deriv_single = std::abs(result_deriv_single(item_id) - reference_deriv(input(item_id), reference(item_id)));
				StandardTestUtils::LogReal("diff_deriv", diff_deriv);
				StandardTestUtils::LogReal("diff_deriv_single",diff_deriv_single);
				Assert::IsTrue(diff_deriv <= 10 * std::numeric_limits<Real>::epsilon(), L"Unexpectedly high deviation from the reference derivative value");
				Assert::IsTrue(diff_deriv_single <= 10 * std::numeric_limits<Real>::epsilon(),
					L"Unexpectedly high deviation from the reference derivative value (single version)");
			}
		}

		/// <summary>
		/// General method to test "CUDA" implementation versus regular (CPU) implementation of the cost function 
		/// </summary>
		static void CheckCudaFunction(const CostFunctionId& func_id)
		{
			//Arrange
			const auto dim = 10;
			const auto input = CudaVector(dim, Real(0.1), Real(0.9));
			const auto reference = CudaVector(dim, Real(0.1), Real(0.9));
			Assert::IsTrue(input != reference, L"Reference and input vectors should not be equal.");

			//Act
			const auto function = CostFunction<CudaVector>(func_id)(input, reference);
			const auto function_and_gradient = CostFunction<CudaVector>(func_id).func_and_deriv(input, reference);
			const auto gradient = CostFunction<CudaVector>(func_id).deriv(input, reference);

			//Assert
			const auto input_host = input.to_host();
			const auto reference_host = reference.to_host();

			const auto function_host = CostFunction<Vector>(func_id)(input_host, reference_host);
			const auto gradient_host = CostFunction<Vector>(func_id).deriv(input_host, reference_host);

			const auto func_diff_1 = std::abs(function - function_host);
			const auto func_diff_2 = std::abs(std::get<0>(function_and_gradient) - function_host);
			const auto gradient_diff_1 = (gradient.to_host() - gradient_host).max_abs();
			const auto gradient_diff_2 = (std::get<1>(function_and_gradient).to_host() - gradient_host).max_abs();
			StandardTestUtils::LogReal("func_diff_1", func_diff_1);
			StandardTestUtils::LogReal("func_diff_2", func_diff_2);
			StandardTestUtils::LogReal("gradient_diff_1", gradient_diff_1);
			StandardTestUtils::LogReal("gradient_diff_2", gradient_diff_2);

			Assert::IsTrue(func_diff_1 < 25 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference (function 1)");
			Assert::IsTrue(func_diff_2 < 25 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference (function 2)");
			Assert::IsTrue(gradient_diff_1 < 10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference (gradient 1)");
			Assert::IsTrue(gradient_diff_2 < 10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference (gradient 2)");
		}

		TEST_METHOD(SquaredErrorFunctionTest)
		{
			CheckFunctionOnly(CostFunctionId::SQUARED_ERROR, SquaredErrorFunc);
		}

		TEST_METHOD(CrossEntropyFunctionTest)
		{
			CheckFunctionOnly(CostFunctionId::CROSS_ENTROPY, CrossEntropyFunc);
		}

		TEST_METHOD(SquaredErrorFunctionAndDerivativeTest)
		{
			CheckFunctionAndDerivative(CostFunctionId::SQUARED_ERROR, SquaredErrorFunc, SquaredErrorDerivFunc);
		}

		TEST_METHOD(CrossEntropyFunctionAndDerivativeTest)
		{
			CheckFunctionAndDerivative(CostFunctionId::CROSS_ENTROPY, CrossEntropyFunc, CrossEntropyDerivFunc);
		}

		TEST_METHOD(SquaredErrorFunctionCudaTest)
		{
			CheckCudaFunction(CostFunctionId::SQUARED_ERROR);
		}

		TEST_METHOD(CrossEntropyFunctionCudaTest)
		{
			CheckCudaFunction(CostFunctionId::CROSS_ENTROPY);
		}
	};
}
