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
#include <Math/Tensor.h>
#include <MsgPackUtils.h>
#include "Utilities.h"
#include "StandardTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(TensorTest)
	{
		TEST_METHOD(TensorCopyConstructorTest)
		{
			//Arrange
			const auto layer_dim = 10;
			const auto row_dim = 13;
			const auto col_dim = 17;
			const auto tensor = Tensor(layer_dim, row_dim, col_dim, -1, 1);

			//Act
			const auto tensor_copy = Tensor(tensor);

			//Assert
			Assert::IsTrue(tensor == tensor_copy, L"Tensors are not the same");
			Assert::IsTrue(tensor.begin() != tensor_copy.begin(), L"Tensors share the same memory");
		}

		TEST_METHOD(TensorCopyAssignmentOperatorTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 13;
			const auto layer_dim = 7;

			const auto row_dim1 = 11;
			const auto col_dim1 = 15;
			const auto layer_dim1 = 17;

			auto tensor_to_assign = Tensor(layer_dim, row_dim, col_dim, -1, 1);
			const auto ptr_before_assignment = tensor_to_assign.begin();

			auto tensor_to_assign1 = Tensor(layer_dim, row_dim, col_dim, -1, 1);
			const auto ptr_before_assignment1 = tensor_to_assign1.begin();

			//Below, layer and row dimensions a swapped on purpose, that way we create
			//a tensor that has different number of layers and rows but its the overall memory
			//footprint is the same as for the tensors above
			const auto tensor_to_copy = Tensor(row_dim, layer_dim, col_dim, -1, 1);
			const auto tensor_to_copy1 = Tensor(layer_dim1, row_dim1, col_dim1, -1, 1);

			Assert::IsTrue(tensor_to_assign != tensor_to_copy && tensor_to_assign != tensor_to_copy1,
				L"Tensors are supposed to be different");

			//Act
			tensor_to_assign = tensor_to_copy;//Assign tensor with the same memory footprint
			tensor_to_assign1 = tensor_to_copy1;//Assign tensor of different memory footprint

			//Assert
			Assert::IsTrue(tensor_to_assign == tensor_to_copy, L"Copying failed (same memory footprint)");
			Assert::IsTrue(ptr_before_assignment == tensor_to_assign.begin(), L"Memory was re-allocated when copying vector of the same memory footprint");

			Assert::IsTrue(tensor_to_assign1 == tensor_to_copy1, L"Copying failed (different memory footprints)");
			Assert::IsTrue(tensor_to_assign1.begin() != tensor_to_copy1.begin(), L"Tensors share the same memory");
		}

		TEST_METHOD(TensorMoveConstructorTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 13;
			const auto layer_dim = 7;

			auto tensor_to_move = Tensor(layer_dim, row_dim, col_dim, -1, 1);
			const auto begin_pointer_before_move = tensor_to_move.begin();
			const auto end_pointer_before_move = tensor_to_move.end();

			//Act
			const Tensor vector(std::move(tensor_to_move));

			//Assert
			Assert::IsTrue(begin_pointer_before_move == vector.begin()
				&& end_pointer_before_move == vector.end(), L"Move operator does not work as expected");

			Assert::IsTrue(tensor_to_move.begin() == nullptr && tensor_to_move.layer_dim() == 0 &&
				tensor_to_move.row_dim() == 0 && tensor_to_move.col_dim() == 0,
				L"Unexpected state for a vector after being moved");
		}

		/// <summary>
		/// Returns random instances of Tensor class of given dimensions
		/// </summary>
		static Tensor TensorFactory(const std::size_t layer_dim = 7, const std::size_t row_dim = 10, const std::size_t col_dim = 13)
		{
			return Tensor(layer_dim, row_dim, col_dim, -1, 1);
		}

		TEST_METHOD(TensorPackingTest)
		{
			StandardTestUtils::PackingTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(SumWithZeroTensorTest)
		{
			const auto row_dim = 10;
			const auto col_dim = 13;
			const auto layer_dim = 7;
			StandardTestUtils::SumWithZeroElementTest<Tensor>([]() { return TensorFactory(layer_dim, row_dim, col_dim); }, Tensor(layer_dim, row_dim, col_dim));
		}

		TEST_METHOD(AdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(DifferenceOfEqualTensorsIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(TensorAdditionAssociativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWuthRespectToTensorAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(TensorMultiplecationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(ConvolutionNoPaddingNoStrideTest)
		{

			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor = TensorFactory(tensor_size.x, tensor_size.y, tensor_size.z);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size.x, kernel_size.y, kernel_size.z);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			const auto result = tensor.convolve(kernel);
			const auto result_size = result.size_3d();

			Assert::IsTrue(result_size.x == tensor_size.x - kernel_size.x + 1 &&
				result_size.y == tensor_size.y - kernel_size.y + 1 &&
				result_size.z == tensor_size.z - kernel_size.z + 1, L"Wrong size of the convolution result");

			for (auto r_l = 0ll; r_l < result_size.x; r_l++)
			{
				for (auto r_r = 0ll; r_r < result_size.y; r_r++)
				{
					for (auto r_c = 0ll; r_c < result_size.z; r_c++)
					{
						Real reference = Real(0);

						for (auto k_l = 0ll; k_l < kernel_size.x; k_l++)
						{
							for (auto k_r = 0ll; k_r < kernel_size.y; k_r++)
							{
								for (auto k_c = 0ll; k_c < kernel_size.z; k_c++)
								{
									reference += kernel(k_l, k_r, k_c) * tensor(r_l + k_l, r_r + k_r, r_c + k_c);
								}
							}
						}

						const auto diff = std::abs(reference - result(r_l, r_r, r_c));
						Logger::WriteMessage((std::string("diff =  ") + Utils::to_string(diff) + "\n").c_str());
						Assert::IsTrue(diff < 10 * std::numeric_limits<Real>::epsilon(), L"Unexpected result");
					}
				}
			}
		}

		TEST_METHOD(ConvolutionWithPaddingNoStrideTest)
		{
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor = TensorFactory(tensor_size.x, tensor_size.y, tensor_size.z);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto padding = Index3d{ 2, 3, 5 };
			//zero tensor
			auto tensor_with_padding = Tensor(tensor_size.x + 2 * padding.x, tensor_size.y + 2 * padding.y, tensor_size.z + 2 * padding.z );

			//copy values to proper location
			for (auto l = 0ll;  l < tensor_size.x; l++ )
				for (auto r = 0ll; r < tensor_size.y; r++)
					for (auto c = 0ll; c < tensor_size.z; c++)
						tensor_with_padding(l + padding.x, r + padding.y, c + padding.z) = tensor(l, r, c);


			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size.x, kernel_size.y, kernel_size.z);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			const auto result = tensor.convolve(kernel, padding);
			const auto result_padding_included = tensor_with_padding.convolve(kernel);

			Assert::IsTrue(result == result_padding_included, L"Tensors are not the same");
		}

		TEST_METHOD(ConvolutionWithStrideNoPaddingTest)
		{
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor = TensorFactory(tensor_size.x, tensor_size.y, tensor_size.z);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto stride = Index3d{ 2, 3, 5 };

			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size.x, kernel_size.y, kernel_size.z);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			const auto result = tensor.convolve(kernel, Index3d{ 0, 0, 0 }, stride);
			const auto result_no_stride = tensor.convolve(kernel);


			for (auto l = 0ul; l < result.layer_dim(); l++)
				for (auto r = 0ul; r < result.row_dim(); r++)
					for (auto c = 0ul; c < result.col_dim(); c++)
					{
						const auto diff = std::abs(result(l, r, c) - result_no_stride(l * stride.x, r * stride.y, c * stride.z));
						Logger::WriteMessage((std::string("diff =  ") + Utils::to_string(diff) + "\n").c_str());
						Assert::IsTrue(diff == Real(0), L"Elements are not the same");
					}
		}
	};
}
