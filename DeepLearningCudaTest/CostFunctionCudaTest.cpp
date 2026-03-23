//Copyright (c) 2026 Denys Dragunov, dragunovdenis@gmail.com
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
	TEST_CLASS(CostFunctionCudaTest)
	{
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
			StandardTestUtils::Log("func_diff_1", func_diff_1);
			StandardTestUtils::Log("func_diff_2", func_diff_2);
			StandardTestUtils::Log("gradient_diff_1", gradient_diff_1);
			StandardTestUtils::Log("gradient_diff_2", gradient_diff_2);

			Assert::IsTrue(func_diff_1 < 25 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference (function 1)");
			Assert::IsTrue(func_diff_2 < 25 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference (function 2)");
			Assert::IsTrue(gradient_diff_1 < 10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference (gradient 1)");
			Assert::IsTrue(gradient_diff_2 < 10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference (gradient 2)");
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
