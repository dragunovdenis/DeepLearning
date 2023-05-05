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
#include <Math/CudaTensor.cuh>
#include <Math/CudaArray.cuh>
#include <Math/ConvolutionUtils.h>
#include <Math/CollectionArithmetics.h>
#include "Math/PoolOperator.h"
#include <Utilities.h>
#include "StandardTestUtils.h"
#include <string>
#include <chrono>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(CudaTensorTest)
	{
		/// <summary>
		/// Returns random instance of CudaTensor
		/// </summary>
		static CudaTensor CudaTensorFactory(const std::size_t layer_dim = 7,
			const std::size_t row_dim = 10, const std::size_t col_dim = 13)
		{
			return CudaTensor(layer_dim, row_dim, col_dim, -1, 1);
		}

		/// <summary>
		/// Returns random instance of CudaTensor
		/// </summary>
		static CudaTensor CudaTensorFactory(const Index3d& size)
		{
			return CudaTensor(size, -1, 1);
		}

		TEST_METHOD(CopyConstructorTest)
		{
			StandardTestUtils::CopyConstructorTest<CudaTensor>([]() {return CudaTensorFactory(); });
		}

		TEST_METHOD(AssignmentOperatorTest)
		{
			StandardTestUtils::AssignmentOperatorTest<CudaTensor>([]() {return CudaTensorFactory(7, 10, 13); },
				[]() {return CudaTensorFactory(13, 11, 25); });
		}

		TEST_METHOD(MoveConstructorTest)
		{
			StandardTestUtils::MoveConstructorTest<CudaTensor>([]() {return CudaTensorFactory(); });
		}

		TEST_METHOD(MoveAssignmentTest)
		{
			StandardTestUtils::MoveAssignmentTest<CudaTensor>([]() {return CudaTensorFactory(); });
		}

		TEST_METHOD(PackingTest)
		{
			StandardTestUtils::PackingTest<CudaTensor>([]() { return CudaTensorFactory(); });
		}

		TEST_METHOD(SumWithZeroVectorTest)
		{
			const auto layer_dim = 7;
			const auto row_dim = 10;
			const auto col_dim = 13;
			StandardTestUtils::SumWithZeroElementTest<CudaTensor>(
				[]() { return CudaTensorFactory(layer_dim, row_dim, col_dim); }, CudaTensor(layer_dim, row_dim, col_dim));
		}

		TEST_METHOD(AdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<CudaTensor>([]() { return CudaTensorFactory(); });
		}

		TEST_METHOD(DifferenceOfEqualVectorsIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<CudaTensor>([]() { return CudaTensorFactory(); });
		}

		TEST_METHOD(AdditionAssociativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<CudaTensor>([]() { return CudaTensorFactory(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWithRespectToVectorAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<CudaTensor>([]() { return CudaTensorFactory(); });
		}

		TEST_METHOD(MultiplicationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<CudaTensor>([]() { return CudaTensorFactory(); });
		}

		TEST_METHOD(MixedArithmeticTest)
		{
			StandardTestUtils::CudaMixedArithmeticTest<CudaTensor>([]() { return CudaTensorFactory(); });
		}

		/// <summary>
		/// Transforms the given "CUDA" 4d tensor into the "host" 4d tensor
		/// </summary>
		std::vector<Tensor> to_host(const std::vector<CudaTensor>& tensor_4d)
		{
			std::vector<Tensor> result;
			std::transform(tensor_4d.begin(), tensor_4d.end(), std::back_inserter(result),
				[](const auto& cudaTensor) { return cudaTensor.to_host(); });

			return result;
		}

		TEST_METHOD(AddScaledTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor1 = CudaTensorFactory(tensor_size);//filled with random numbers
			const auto tensor2 = CudaTensorFactory(tensor_size);//filled with random numbers
			const auto scalar = Utils::get_random(1, 5);
			Assert::IsTrue(tensor1.max_abs() > 0 && tensor2.max_abs() > 0,
				L"Tensors are supposed to be nonzero");

			//Act
			auto result1 = tensor1;
			result1.add_scaled(tensor2, scalar);
			const auto result2 = tensor1 + tensor2 * scalar;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() <
				10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference");
		}

		TEST_METHOD(ScaleAndAddTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor1 = CudaTensorFactory(tensor_size);//filled with random numbers
			const auto tensor2 = CudaTensorFactory(tensor_size);//filled with random numbers
			const auto scalar = Utils::get_random(1, 5);
			Assert::IsTrue(tensor1.max_abs() > 0 && tensor2.max_abs() > 0,
				L"Tensors are supposed to be nonzero");

			//Act
			auto result1 = tensor1;
			result1.scale_and_add(tensor2, scalar);
			const auto result2 = tensor1 * scalar + tensor2;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() <
				10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference");
		}

		TEST_METHOD(ScaleAndAddScaledTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor1 = CudaTensorFactory(tensor_size);//filled with random numbers
			const auto tensor2 = CudaTensorFactory(tensor_size);//filled with random numbers
			const auto scalar_0 = Utils::get_random(1, 5);
			const auto scalar_1 = Utils::get_random(1, 5);
			Assert::IsTrue(tensor1.max_abs() > 0 && tensor2.max_abs() > 0,
				L"Tensors are supposed to be nonzero");

			//Act
			auto result1 = tensor1;
			result1.scale_and_add_scaled(scalar_0, tensor2, scalar_1);
			const auto result2 = tensor1 * scalar_0 + tensor2 * scalar_1;
			
			//Assert
			Assert::IsTrue((result1 - result2).max_abs() <
				10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference");
		}

		TEST_METHOD(FourDimTensorSumTest)
		{
			//Arrange
			const auto tensor_4d_1 = std::vector<CudaTensor>{ CudaTensorFactory() , CudaTensorFactory() , CudaTensorFactory() };
			const auto tensor_4d_2 = std::vector<CudaTensor>{ CudaTensorFactory() , CudaTensorFactory() , CudaTensorFactory() };

			//Act
			auto result = tensor_4d_1;
			result += tensor_4d_2;

			//Assert
			auto result_host = to_host(tensor_4d_1);
			result_host += to_host(tensor_4d_2);
			Assert::IsTrue(result_host == to_host(result));
		}

		TEST_METHOD(FourDimTensorScaleTest)
		{
			//Arrange
			const auto tensor_4d = std::vector<CudaTensor>{ CudaTensorFactory() , CudaTensorFactory() , CudaTensorFactory() };
			const auto scalar = Utils::get_random(1, 10);

			//Act
			auto result = tensor_4d;
			result *= scalar;

			//Assert
			auto result_host = to_host(tensor_4d);
			result_host *= scalar;
			Assert::IsTrue(result_host == to_host(result));
		}

		TEST_METHOD(ConvolutionTest)
		{
			//Arrange
			const CudaTensor tensor(20, 128, 128, -1, 1);
			const CudaTensor kernel(11, 5, 5, -1, 1);
			Assert::IsTrue(tensor.max_abs() > 0, L"The tensor is supposed to be non-zero");
			Assert::IsTrue(kernel.max_abs() > 0, L"The kernel is supposed to be non-zero");
			const Index3d paddings = { 0, 1, 2 };
			const Index3d strides = { 1, 2, 3 };

			//Act
			const auto result = tensor.convolve(kernel, paddings, strides);

			//Assert
			const auto result_reference_host = tensor.to_host().convolve(kernel.to_host(), paddings, strides);
			const auto diff = (result.to_host() - result_reference_host).max_abs();
			Logger::WriteMessage((std::string("Diff = ") + Utils::to_string(diff)).c_str());
			Assert::IsTrue(diff < 100 * std::numeric_limits<Real>::epsilon(), L"Unexpectedly high deviation from reference");
		}

		TEST_METHOD(ConvolutionWithCollectionOfKernelsTest)
		{
			//Arrange
			const CudaTensor tensor(20, 128, 128, -1, 1);
			Assert::IsTrue(tensor.max_abs() > 0, L"Input tensor is supposed to be nonzero");
			std::vector<CudaTensor> kernels(10, { 20, 5, 5 });
			const Index3d paddings = { 0, 1, 2 };
			const Index3d strides = { 1, 2, 3 };

			for (auto kernel_id = 0; kernel_id < 10; kernel_id++)
			{
				kernels[kernel_id].uniform_random_fill(-1, 1);
				Assert::IsTrue(kernels[kernel_id].max_abs() > 0, L"Kernels are supposed to be nonzero");
			}
			const auto channel_size = ConvolutionUtils::calc_conv_res_size(tensor.size_3d(), kernels[0].size_3d(), paddings, strides);
			Assert::IsTrue(channel_size.x == 1, L"Unexpected number of layers in a single channel");
			CudaTensor result(kernels.size(), channel_size.y, channel_size.z, false);

			//Act
			tensor.convolve(result, kernels, paddings, strides);

			//Assert
			std::vector<Tensor> kernels_host;
			for (const auto& kernel : kernels)
				kernels_host.push_back(kernel.to_host());

			Tensor result_host(kernels.size(), channel_size.y, channel_size.z, false);
			const auto tensor_host = tensor.to_host();
			tensor_host.convolve(result_host, kernels_host, paddings, strides);

			const auto diff = (result.to_host() - result_host).max_abs();

			Logger::WriteMessage((std::string("Diff = ") + Utils::to_string(diff)).c_str());
			Assert::IsTrue(diff < 200 * std::numeric_limits<Real>::epsilon(), L"Unexpectedly high deviation from reference");
		}

		TEST_METHOD(ConvolutionGradientTest)
		{
			//Arrange
			const CudaTensor tensor(10, 28, 30, -1, 1);
			const CudaTensor kernel(10, 5, 4, -1, 1);
			const Index3d paddings = { 0, 1, 2 };
			const Index3d strides = { 1, 2, 3 };
			const CudaTensor res_grad(ConvolutionUtils::calc_conv_res_size(tensor.size_3d(), kernel.size_3d(), paddings, strides), -1, 1);
			Assert::IsTrue(tensor.max_abs() > 0, L"The tensor is supposed to be non-zero");
			Assert::IsTrue(res_grad.max_abs() > 0, L"The convolution gradient is supposed to be non-zero");
			Assert::IsTrue(kernel.max_abs() > 0, L"The kernel is supposed to be non-zero");

			//Act
			const auto [kernel_grad, input_grad] = tensor.convolution_gradient(res_grad, kernel, paddings, strides);

			//Assert
			const auto [kernel_grad_host, input_grad_host] = tensor.to_host().convolution_gradient(res_grad.to_host(), kernel.to_host(), paddings, strides);

			const auto kernel_grad_diff = (kernel_grad.to_host() - kernel_grad_host).max_abs();
			const auto input_grad_diff = (input_grad.to_host() - input_grad_host).max_abs();

			Logger::WriteMessage((std::string("kernel_grad_diff= ") + Utils::to_string(kernel_grad_diff) + "\n").c_str());
			Logger::WriteMessage((std::string("input_grad_diff= ") + Utils::to_string(input_grad_diff) + "\n").c_str());

			Assert::IsTrue(kernel_grad_diff < 100 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference for the kernel gradient");
			Assert::IsTrue(input_grad_diff < 50 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference for the input gradient");
		}

		void min_max_pool_optimixed_test(const bool max)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 11, 9 };
			const auto tensor = CudaTensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");
			const auto pool_window_size = Index3d(2, 3, 4);
			const CudaTensor res_grad(ConvolutionUtils::calc_conv_res_size(tensor.size_3d(), pool_window_size, { 0 }, pool_window_size), -1, 1);
			Assert::IsTrue(res_grad.max_abs() > 0, L"Gradient is expected to be nonzero");

			//Act
			const auto [result, mapping] = tensor.min_max_pool(pool_window_size, max);
			const auto gradient = tensor.min_max_pool_input_gradient(res_grad, mapping);

			//Assert
			const auto [result_host, mapping_host] = tensor.to_host().min_max_pool(pool_window_size, max);
			const auto gradient_host = tensor.to_host().min_max_pool_input_gradient(res_grad.to_host(), mapping.to_stdvector());

			Assert::IsTrue(result.to_host() == result_host, L"Unexpected result of pooling operation");
			Assert::IsTrue(mapping.to_stdvector() == mapping_host, L"Unexpected mapping of pooling operation");
			Assert::IsTrue(gradient.to_host() == gradient_host, L"Unexpected gradient of pooling operation");
		}

		TEST_METHOD(MinPoolOptimizedTest)
		{
			min_max_pool_optimixed_test(/*max*/ false);
		}

		TEST_METHOD(MaxPoolOptimizedTest)
		{
			min_max_pool_optimixed_test(/*max*/ true);
		}

		TEST_METHOD(AveragePoolTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 11, 9 };
			const auto tensor = CudaTensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");
			const auto pool_window_size = Index3d(2, 3, 4);
			const CudaTensor res_grad(ConvolutionUtils::calc_conv_res_size(tensor.size_3d(), pool_window_size, { 0 }, pool_window_size), -1, 1);
			Assert::IsTrue(res_grad.max_abs() > 0, L"Gradient is expected to be nonzero");

			//Act
			const auto result = tensor.average_pool(pool_window_size);
			const auto gradient = tensor.average_pool_input_gradient(res_grad, pool_window_size);

			//Assert
			const auto result_host = tensor.to_host().average_pool(pool_window_size);
			const auto gradient_host = tensor.to_host().average_pool_input_gradient(res_grad.to_host(), pool_window_size);

			const auto result_diff = (result.to_host() - result_host).max_abs();
			const auto gradient_diff = (gradient.to_host() - gradient_host).max_abs();

			Logger::WriteMessage((std::string("Difference for result = ") + Utils::to_string(result_diff) + "\n").c_str());
			Logger::WriteMessage((std::string("Difference for gradient = ") + Utils::to_string(gradient_diff) + "\n").c_str());

			Assert::IsTrue(result.to_host() == result_host, L"Unexpected result of pooling operation");
			Assert::IsTrue(gradient.to_host() == gradient_host, L"Unexpected gradient of pooling operation");
		}
	};
}