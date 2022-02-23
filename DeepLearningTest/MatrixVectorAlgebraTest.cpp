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
#include <Math/DenseMatrix.h>
#include <Math/DenseVector.h>
#include <MsgPackUtils.h>
#include "Utilities.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(MatrixVectorAlgebraTest)
	{
	public:

		TEST_METHOD(VectorCopyConstructorTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector = DenseVector(dim, -1, 1);

			//Act
			const auto vector_copy = DenseVector(vector);

			//Assert
			Assert::IsTrue(vector == vector_copy, L"Vectors are not the same");
			Assert::IsTrue(vector.begin() != vector_copy.begin(), L"Vectors share the same memory");
		}

		TEST_METHOD(VectorCopyAssignmentOperatorTest)
		{
			//Arrange
			const auto dim = 10;
			const auto dim1 = 13;
			auto vec_to_assign = DenseVector(dim, -1, 1);
			const auto ptr_before_assignment = vec_to_assign.begin();

			auto vec_to_assign1 = DenseVector(dim, -1, 1);
			const auto ptr_before_assignment1 = vec_to_assign1.begin();

			const auto vec_to_copy = DenseVector(dim, -1, 1);
			const auto vec_to_copy1 = DenseVector(dim1, -1, 1);

			Assert::IsTrue(vec_to_assign != vec_to_copy && vec_to_assign != vec_to_copy1,
				L"Vectors are supposed to be different");

			//Act
			vec_to_assign = vec_to_copy;//Assign vector of the same size
			vec_to_assign1 = vec_to_copy1;//Assign vector of different size

			//Assert
			Assert::IsTrue(vec_to_assign == vec_to_copy, L"Copying failed (same size)");
			Assert::IsTrue(ptr_before_assignment == vec_to_assign.begin(), L"Memory was re-allocated when copying vector of the same size");

			Assert::IsTrue(vec_to_assign1 == vec_to_copy1, L"Copying failed (different sizes)");
			Assert::IsTrue(vec_to_assign1.begin() != vec_to_copy1.begin(), L"Vectors share the same memory");
		}

		TEST_METHOD(VectorMoveConstructorTest)
		{
			//Arrange
			const auto dim = 10;
			auto vector_to_move = DenseVector(dim, -1, 1);
			const auto begin_pointer_before_move = vector_to_move.begin();
			const auto end_pointer_before_move = vector_to_move.end();

			//Act
			const DenseVector vector(std::move(vector_to_move));

			//Assert
			Assert::IsTrue(begin_pointer_before_move == vector.begin()
				&& end_pointer_before_move == vector.end(), L"Move operator does not work as expected");

			Assert::IsTrue(vector_to_move.begin() == nullptr && vector_to_move.dim() == 0,
				L"Unexpected state for a vector after being moved");
		}

		TEST_METHOD(MatrixCopyConstructorTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 13;
			const auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);

			//Act
			const auto matrix_copy = DenseMatrix(matrix);

			//Assert
			Assert::IsTrue(matrix == matrix_copy, L"Matrices are not the same");
			Assert::IsTrue(matrix.begin() != matrix_copy.begin(), L"Matrices share the same memory");
		}

		TEST_METHOD(MatrixCopyAssignmentOperatorTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 13;
			const auto row_dim1 = 20;
			const auto col_dim1 = 23;
			auto matr_to_assign = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto ptr_before_assignment = matr_to_assign.begin();

			auto matr_to_assign1 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto ptr_before_assignment1 = matr_to_assign1.begin();

			const auto matr_to_copy = DenseMatrix(col_dim, row_dim, -1, 1);
			const auto matr_to_copy1 = DenseMatrix(row_dim1, col_dim1, -1, 1);

			Assert::IsTrue(matr_to_assign != matr_to_copy && matr_to_assign != matr_to_copy1,
				L"Matrices are supposed to be different");

			//Act
			matr_to_assign = matr_to_copy;//Assign matrix of the same size
			matr_to_assign1 = matr_to_copy1;//Assign matrix of different size

			//Assert
			Assert::IsTrue(matr_to_assign == matr_to_copy, L"Copying failed (same size)");
			Assert::IsTrue(ptr_before_assignment == matr_to_assign.begin(), L"Memory was re-allocated when copying matrix of the same size");

			Assert::IsTrue(matr_to_assign1 == matr_to_copy1, L"Copying failed (different sizes)");
			Assert::IsTrue(matr_to_assign1.begin() != matr_to_copy1.begin(), L"Matrices share the same memory");
		}

		TEST_METHOD(MatrixMoveConstructorTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 13;
			auto matrix_to_move = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto begin_pointer_before_move = matrix_to_move.begin();
			const auto end_pointer_before_move = matrix_to_move.end();

			//Act
			const DenseMatrix matrix(std::move(matrix_to_move));

			//Assert
			Assert::IsTrue(begin_pointer_before_move == matrix.begin()
				&& end_pointer_before_move == matrix.end(), L"Move operator does not work as expected");

			Assert::IsTrue(matrix_to_move.begin() == nullptr &&
				matrix_to_move.row_dim() == 0 && matrix_to_move.col_dim() == 0,
				L"Unexpected state for a matrix after being moved");
		}

		TEST_METHOD(MatrixVectorMultiplicationTest)
		{
			//Arrange
			const std::size_t row_dim = 10;
			const std::size_t col_dim = 23;
			const auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto vector = DenseVector(col_dim, -1, 1);

			//Act
			const auto product = matrix * vector;

			//Assert
			for (std::size_t row_id = 0; row_id < row_dim; row_id++)
			{
				Real reference = Real(0);

				for (std::size_t col_id = 0; col_id < col_dim; col_id++)
				{
					reference += matrix(row_id, col_id) * vector(col_id);
				}

				const auto diff = std::abs(reference - product(row_id));
				Logger::WriteMessage((std::string("Difference = ") +  Utils::to_string(diff) + "\n").c_str());
				Assert::IsTrue(diff < 15*std::numeric_limits<Real>::epsilon(), L"Unexpectedly high difference");
			}
		}

		TEST_METHOD(VectorMatrixMultiplicationTest)
		{
			//Arrange
			const std::size_t row_dim = 43;
			const std::size_t col_dim = 14;
			const auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto vector = DenseVector(row_dim, -1, 1);

			//Act
			const auto product = vector * matrix;

			//Assert
			for (std::size_t col_id = 0; col_id < col_dim; col_id++)
			{
				Real reference = Real(0);

				for (std::size_t row_id = 0; row_id < row_dim; row_id++)
				{
					reference += matrix(row_id, col_id) * vector(row_id);
				}

				const auto diff = std::abs(reference - product(col_id));
				Logger::WriteMessage((std::string("Difference = ") + std::to_string(diff) + "\n").c_str());
				Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(), L"Unexpectedly high difference.");
			}
		}

		TEST_METHOD(VectorPackingTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector = DenseVector(dim, -1, 1);

			//Act
			const auto msg = MsgPack::pack(vector);
			const auto vector_unpacked = MsgPack::unpack<DenseVector>(msg);

			//Assert
			Assert::IsTrue(vector == vector_unpacked, L"De-serialized vector is not equal to the original one.");
		}

		TEST_METHOD(MatrixPackingTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 33;
			const auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);

			//Act
			const auto msg = MsgPack::pack(matrix);
			const auto matrix_unpacked = MsgPack::unpack<DenseMatrix>(msg);

			//Assert
			Assert::IsTrue(matrix == matrix_unpacked, L"De-serialized matrix is not equal to the original one.");
		}

		TEST_METHOD(SumWithZeroVectorTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector = DenseVector(dim, -1, 1);
			Assert::IsTrue(vector.max_abs() > 0, L"The vector is zero!");
			const auto zero_vector = DenseVector(dim);
			Assert::IsTrue(zero_vector.max_abs() == 0, L"The vector is non-zero!");

			//Act
			const auto result = vector + zero_vector;

			//Assert
			Assert::IsTrue(vector == result, L"Vectors are not equal!");
		}

		TEST_METHOD(VectorAdditionCommutativityTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector1 = DenseVector(dim, -1, 1);
			const auto vector2 = DenseVector(dim, -1, 1);
			Assert::IsTrue(vector1.max_abs() > 0 && vector2.max_abs() > 0, L"The input vectors are expected to be non-zero!");
			Assert::IsTrue(vector1 != vector2, L"The input vectors are supposed to be different!");

			//Act
			const auto result1 = vector1 + vector2;
			const auto result2 = vector2 + vector1;

			//Assert
			Assert::IsTrue(result1 == result2, L"Vector addition operator is non-commutative!");
		}

		TEST_METHOD(DifferenceOfEqualVectorsIsZeroTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector1 = DenseVector(dim, -1, 1);
			const auto vector2 = vector1;
			Assert::IsTrue(vector1.max_abs() > 0, L"The input vector is expected to be non-zero!");
			Assert::IsTrue(vector1 == vector2, L"The input vectors are not equal!");

			//Act
			const auto result = vector1 - vector2;

			//Assert
			Assert::IsTrue(result.max_abs() == 0, L"The result is non-zero");
		}

		TEST_METHOD(VectorAdditionAssocoativityTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector1 = DenseVector(dim, -1, 1);
			const auto vector2 = DenseVector(dim, -1, 1);
			const auto vector3 = DenseVector(dim, -1, 1);
			Assert::IsTrue(vector1.max_abs() > 0 &&
							vector2.max_abs() > 0 &&
							vector3.max_abs() > 0, L"The input vectors are expected to be non-zero!");
			Assert::IsTrue(vector1 != vector2 && vector1 != vector3 && vector3 != vector2, L"The input vectors are supposed to be different!");

			//Act
			const auto result1 = (vector1 + vector2) + vector3;
			const auto result2 = vector1 + (vector2 + vector3);

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Vector addition operator is non-associative!");
		}

		TEST_METHOD(DistributivityOfVectorAdditionWithRespectToScalarMultiplicationTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector1 = DenseVector(dim, -1, 1);
			const auto vector2 = DenseVector(dim, -1, 1);
			const auto scalar = DenseVector(1, -1, 1)(0);
			Assert::IsTrue(vector1.max_abs() > 0 && vector2.max_abs() > 0, L"The input vectors are expected to be non-zero!");
			Assert::IsTrue(scalar != 0, L"Scalar is expected to be non-zero!");

			//Act
			const auto result1 = (vector1 + vector2) * scalar;
			const auto result2 = vector1*scalar + vector2*scalar;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Vector addition operator is non-distributive with respect to scalar multiplication!");
		}

		TEST_METHOD(VectorMultiplicationByOneTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector = DenseVector(dim, -1, 1);
			const auto scalar = Real(1);
			Assert::IsTrue(vector.max_abs() > 0, L"The input vector is expected to be non-zero!");

			//Act
			const auto result = vector * scalar;

			//Assert
			Assert::IsTrue(result == vector, L"Vectors are not the same!");
		}

		TEST_METHOD(SumWithZeroMatrixTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto zero_matrix = DenseMatrix(row_dim, col_dim);
			Assert::IsTrue(matrix.max_abs() > 0, L"The matrix is supposed to be non-zero!");
			Assert::IsTrue(zero_matrix.max_abs() == 0, L"Zero matrix is actually non-zero!");

			//Act
			const auto result = matrix + zero_matrix;

			//Assert
			Assert::IsTrue(result == matrix, L"Matrices are not the same!");
		}

		TEST_METHOD(MatrixAdditionCommutativityTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix1 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto matrix2 = DenseMatrix(row_dim, col_dim, -1, 1);
			Assert::IsTrue(matrix1.max_abs() > 0 && matrix2.max_abs() > 0, L"The input matrices are supposed to be non-zero!");
			Assert::IsTrue(matrix1 != matrix2, L"The input matrices must not be equal!");

			//Act
			const auto result1 = matrix1 + matrix2;
			const auto result2 = matrix2 + matrix1;

			//Assert
			Assert::IsTrue(result1 == result2, L"The matrix addition operator is non-commutative!");
		}

		TEST_METHOD(differenceOfEqualMatricesIsZeroTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix1 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto matrix2 = matrix1;
			Assert::IsTrue(matrix1.max_abs() > 0 && matrix2.max_abs() > 0,
				L"The input matrices are supposed to be non-zero!");
			Assert::IsTrue(matrix1 == matrix2, L"The input matrices must be equal!");

			//Act
			const auto result = matrix1 - matrix2;

			//Assert
			Assert::IsTrue(result.max_abs() == 0, L"Result is supposed to be zeroA");
		}

		TEST_METHOD(MatrixAdditionAssociativityTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix1 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto matrix2 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto matrix3 = DenseMatrix(row_dim, col_dim, -1, 1);
			Assert::IsTrue(matrix1.max_abs() > 0 && matrix2.max_abs() > 0 && matrix3.max_abs() > 0,
				L"The input matrices are supposed to be non-zero!");
			Assert::IsTrue(matrix1 != matrix2 && matrix1 != matrix3 && matrix3 != matrix2,
				L"The input matrices must not be equal!");

			//Act
			const auto result1 = (matrix1 + matrix2) + matrix3;
			const auto result2 = matrix1 + (matrix2 + matrix3);

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Matrix addition operator is non-associative");
		}

		TEST_METHOD(DistributivityOfMatrixAdditionWithRespectToScalarMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix1 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto matrix2 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto scalar = DenseMatrix(1, 1, -1, 1)(0, 0);
			Assert::IsTrue(matrix1.max_abs() > 0 && matrix2.max_abs() > 0,
				L"The input matrices are supposed to be non-zero!");
			Assert::IsTrue(std::abs(scalar) > 0, L"Scalar is supposed to be non-zero!");

			//Act
			const auto result1 = (matrix1 + matrix2) * scalar;
			const auto result2 = matrix1 * scalar + matrix2 * scalar;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Matrix addition is non-distributable with respect to scalar multiplication");
		}

		TEST_METHOD(MatrixMultiplicationByOneTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto scalar = Real(1);
			Assert::IsTrue(matrix.max_abs() > 0,
				L"The input matrix is supposed to be non-zero!");

			//Act
			const auto result = matrix * scalar;

			//Assert
			Assert::IsTrue(matrix == result, L"Matrices are supposed to be equal!");
		}

		TEST_METHOD(DistributivityOfMatrixAdditionWithRespectToRightVectorMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix1 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto matrix2 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto vector = DenseVector(col_dim, -1, 1);
			Assert::IsTrue(matrix1.max_abs() > 0 && matrix2.max_abs() > 0,
				L"The input matrices are supposed to be non-zero!");
			Assert::IsTrue(vector.max_abs() > 0, L"The input vector is supposed to be non-zero!");

			//Act
			const auto result1 = (matrix1 + matrix2) * vector;
			const auto result2 = matrix1 * vector + matrix2 * vector;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10*std::numeric_limits<Real>::epsilon(),
				L"Matrix addition is non-distributive with respect to vector multiplication from the right.");
		}

		TEST_METHOD(DistributivityOfMatrixAdditionWithRespectToLeftVectorMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix1 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto matrix2 = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto vector = DenseVector(row_dim, -1, 1);
			Assert::IsTrue(matrix1.max_abs() > 0 && matrix2.max_abs() > 0,
				L"The input matrices are supposed to be non-zero!");
			Assert::IsTrue(vector.max_abs() > 0, L"The input vector is supposed to be non-zero!");

			//Act
			const auto result1 = vector * (matrix1 + matrix2);
			const auto result2 = vector * matrix1  + vector * matrix2;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Matrix addition is non-distributive with respect to vector multiplication from the left.");
		}

		TEST_METHOD(DistributivityOfVectorAdditionWithRespectToLeftMatrixMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto vector1 = DenseVector(col_dim, -1, 1);
			const auto vector2 = DenseVector(col_dim, -1, 1);
			Assert::IsTrue(matrix.max_abs() > 0,
				L"The input matrix is supposed to be non-zero!");
			Assert::IsTrue(vector1.max_abs() > 0 && vector2.max_abs() > 0,
				L"The input vectors are supposed to be non-zero!");

			//Act
			const auto result1 = matrix * (vector1 + vector2);
			const auto result2 = matrix * vector1  + matrix * vector2;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Vector addition is non-distributive with respect to matrix multiplication from the left.");
		}

		TEST_METHOD(DistributivityOfVectorAdditionWithRespectToRightMatrixMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto vector1 = DenseVector(row_dim, -1, 1);
			const auto vector2 = DenseVector(row_dim, -1, 1);
			Assert::IsTrue(matrix.max_abs() > 0,
				L"The input matrix is supposed to be non-zero!");
			Assert::IsTrue(vector1.max_abs() > 0 && vector2.max_abs() > 0,
				L"The input vectors are supposed to be non-zero!");

			//Act
			const auto result1 = (vector1 + vector2) * matrix;
			const auto result2 = vector1 * matrix + vector2 * matrix;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Vector addition is non-distributive with respect to matrix multiplication from the right.");
		}

		TEST_METHOD(VectorColByVectorRowMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto vec_col = DenseVector(row_dim, -1, 1);
			const auto vec_row = DenseVector(col_dim, -1, 1);
			Assert::IsTrue(vec_col.max_abs() > 0 && vec_row.max_abs() > 0,
				L"The input vectors are supposed to be non-zero!");

			//Act
			const auto result = vector_col_times_vector_row(vec_col, vec_row);

			//Assert
			Assert::IsTrue(result.col_dim() == col_dim && result.row_dim() == row_dim, L"Unexpected dimensions of the resulting matrix.");

			for (std::size_t row_id = 0; row_id < row_dim; row_id++)
			{
				for (std::size_t col_id = 0; col_id < col_dim; col_id++)
				{
					const auto diff = std::abs(result(row_id, col_id) - vec_col(row_id) * vec_row(col_id));
					Assert::IsTrue(diff <= 0, L"Too big deviation from expected value");
				}
			}
		}

		TEST_METHOD(HadamardVectorProductTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector1 = DenseVector(dim, -1, 1);
			const auto vector2 = DenseVector(dim, -1, 1);
			Assert::IsTrue(vector1.max_abs() > 0 && vector2.max_abs() > 0,
				L"The input vectors are supposed to be non-zero!");

			//Act
			const auto result1 = vector1.hadamard_prod(vector2);
			const auto result2 = vector2.hadamard_prod(vector1);

			//Assert
			Assert::IsTrue(result1 == result2, L"The Hadamard product should be commutative.");

			for (std::size_t item_id = 0; item_id < dim; item_id++)
			{
				const auto diff = std::abs(result1(item_id) - vector1(item_id) * vector2(item_id));
				Assert::IsTrue(diff <= 0, L"Too big deviation from the expected value.");
			}
		}

		TEST_METHOD(MatrixVectorMultiplicationAndAdditionTest)
		{
			//Arrange
			const std::size_t row_dim = 10;
			const std::size_t col_dim = 23;
			const auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);
			const auto mul_vector = DenseVector(col_dim, -1, 1);
			const auto add_vector = DenseVector(row_dim, -1, 1);

			//Act
			const auto result1 = matrix * mul_vector + add_vector;
			const auto result2 = matrix.mul_add(mul_vector, add_vector);

			//Assert
			const auto diff = (result1 - result2).max_abs();
			Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + "\n").c_str());
			Assert::IsTrue(diff < 10 * std::numeric_limits<Real>::epsilon(), L"Unexpectedly high difference");
		}
	};
}
