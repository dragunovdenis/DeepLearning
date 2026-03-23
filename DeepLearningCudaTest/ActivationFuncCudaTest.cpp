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
#include <Math/ActivationFunction.h>
#include <Utilities.h>
#include <numeric>
#include "StandardTestUtils.h"
#include "ActivationFuncTestUtils.h"
#include "Math/CudaTensor.cuh"
#include "Math/Functions.h"
#include "Math/Tensor.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(ActivationFunctionCudaTest)
	{
	public:
		TEST_METHOD(SigmoidFunctionCudaTest)
		{
			//Arrange
			constexpr auto dim = 10000;
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

		TEST_METHOD(SigmoidFunctionAndDerivativeCudaTest)
		{
			//Arrange
			constexpr auto dim = 10;
			const auto vector = CudaVector(dim, -1, 1);
			const auto out_grad = CudaVector(dim, -1, 1);
			Assert::IsTrue(vector.max_abs() > 0 && out_grad.max_abs() > 0, L"Vectors are supposed to be nonzero");

			//Act
			const auto function = ActivationFunction<CudaVector>(ActivationFunctionId::SIGMOID);
			const auto [result, result_aux] = function.func_and_aux(vector);
			const auto result_gradient = function.get_in_grad(out_grad, result_aux);

			//Assert
			const auto function_host = ActivationFunction<Vector>(ActivationFunctionId::SIGMOID);
			const auto [result_host, result_aux_host] = function_host.func_and_aux(vector.to_host());
			const auto result_gradient_host = function_host.get_in_grad(out_grad.to_host(), result_aux.to_host());

			Assert::IsTrue((result_host - result.to_host()).max_abs() < 10 * std::numeric_limits<Real>::epsilon(), 
				L"Result: too high deviation from reference");
			Assert::IsTrue((result_aux_host - result_aux.to_host()).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Auxiliary data: Too high deviation from reference");
			Assert::IsTrue((result_gradient_host - result_gradient.to_host()).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Gradient: Too high deviation from reference");
		}

		TEST_METHOD(SoftMaxFunctionCudaTest)
		{
			//Arrange
			constexpr auto dim = 10;
			const auto vec = CudaVector(dim, -1, 1);
			Assert::IsTrue(vec.max_abs() > 0, L"Vector is supposed to be nonzero");

			//Act
			const auto soft_max_result = SoftMaxActivationFunction<CudaVector>()(vec);

			//Assert
			const auto soft_max_result_host = SoftMaxActivationFunction<Vector>()(vec.to_host());
			const auto diff = (soft_max_result.to_host() - soft_max_result_host).max_abs();
			Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(), L"Too high deviation between the actual and expected values");
		}

		TEST_METHOD(SoftMaxFunctionAndDerivativeCudaTest)
		{
			//Arrange
			constexpr auto dim = 10;
			const auto input = CudaVector(dim, -1, 1);
			const auto out_grad = CudaVector(dim, -1, 1);
			Assert::IsTrue(input.max_abs() > 0 && out_grad.max_abs() > 0, L"Vectors are supposed to be nonzero");

			//Act
			const auto [soft_max_result, aux_data] = SoftMaxActivationFunction<CudaVector>().func_and_aux(input);
			const auto activation_input_gradient = SoftMaxActivationFunction<CudaVector>().get_in_grad(out_grad, aux_data);

			//Assert
			const auto [soft_max_result_host, aux_data_host] = SoftMaxActivationFunction<Vector>().func_and_aux(input.to_host());
			const auto activation_input_gradient_host = SoftMaxActivationFunction<Vector>().get_in_grad(out_grad.to_host(), aux_data.to_host());
			const auto diff_func = (soft_max_result.to_host() - soft_max_result_host).max_abs();
			const auto diff_grad = (activation_input_gradient.to_host() - activation_input_gradient_host).max_abs();
			Assert::IsTrue(diff_func < std::numeric_limits<Real>::epsilon(), L"Too high deviation between the actual and expected values (function)");
			Assert::IsTrue(diff_grad < std::numeric_limits<Real>::epsilon(), L"Too high deviation between the actual and expected values (gradient)");
		}

		TEST_METHOD(ReLuOptimizedVsGeneralCudaTest)
		{
			RunOptimizedVsGeneralActivationFuncTest(ReLuActivationFunction<CudaTensor>(), ActivationFunctionId::RELU);
		}

		TEST_METHOD(SigmoidOptimizedVsGeneralCudaTest)
		{
			RunOptimizedVsGeneralActivationFuncTest(SigmoidActivationFunction<CudaTensor>(), ActivationFunctionId::SIGMOID);
		}

		TEST_METHOD(TanhOptimizedVsGeneralCudaTest)
		{
			RunOptimizedVsGeneralActivationFuncTest(TanhActivationFunction<CudaTensor>(), ActivationFunctionId::TANH);
		}
	};
}
