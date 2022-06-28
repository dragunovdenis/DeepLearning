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
	};
}