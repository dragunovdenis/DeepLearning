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
#include <Math/DenseTensor.h>
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
			const auto tensor = DenseTensor(layer_dim, row_dim, col_dim, -1, 1);

			//Act
			const auto tensor_copy = DenseTensor(tensor);

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

			auto tensor_to_assign = DenseTensor(layer_dim, row_dim, col_dim, -1, 1);
			const auto ptr_before_assignment = tensor_to_assign.begin();

			auto tensor_to_assign1 = DenseTensor(layer_dim, row_dim, col_dim, -1, 1);
			const auto ptr_before_assignment1 = tensor_to_assign1.begin();

			//Below, layer and row dimensions a swapped on purpose, that way we create
			//a tensor that has different number of layers and rows but its the overall memory
			//footprint is the same as for the tensors above
			const auto tensor_to_copy = DenseTensor(row_dim, layer_dim, col_dim, -1, 1);
			const auto tensor_to_copy1 = DenseTensor(layer_dim1, row_dim1, col_dim1, -1, 1);

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

			auto tensor_to_move = DenseTensor(layer_dim, row_dim, col_dim, -1, 1);
			const auto begin_pointer_before_move = tensor_to_move.begin();
			const auto end_pointer_before_move = tensor_to_move.end();

			//Act
			const DenseTensor vector(std::move(tensor_to_move));

			//Assert
			Assert::IsTrue(begin_pointer_before_move == vector.begin()
				&& end_pointer_before_move == vector.end(), L"Move operator does not work as expected");

			Assert::IsTrue(tensor_to_move.begin() == nullptr && tensor_to_move.layer_dim() == 0 &&
				tensor_to_move.row_dim() == 0 && tensor_to_move.col_dim() == 0,
				L"Unexpected state for a vector after being moved");
		}

		/// <summary>
		/// Returns random instances of DenseTensor class of given dimensions
		/// </summary>
		static DenseTensor TensorFactory(const std::size_t layer_dim = 7, const std::size_t row_dim = 10, const std::size_t col_dim = 13)
		{
			return DenseTensor(layer_dim, row_dim, col_dim, -1, 1);
		}

		TEST_METHOD(TensorPackingTest)
		{
			StandardTestUtils::PackingTest<DenseTensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(SumWithZeroTensorTest)
		{
			const auto row_dim = 10;
			const auto col_dim = 13;
			const auto layer_dim = 7;
			StandardTestUtils::SumWithZeroElementTest<DenseTensor>([]() { return TensorFactory(layer_dim, row_dim, col_dim); }, DenseTensor(layer_dim, row_dim, col_dim));
		}

		TEST_METHOD(AdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<DenseTensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(DifferenceOfEqualTensorsIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<DenseTensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(TensorAdditionAssociativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<DenseTensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWuthRespectToTensorAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<DenseTensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(TensorMultiplecationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<DenseTensor>([]() { return TensorFactory(); });
		}
	};
}
