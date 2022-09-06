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
#include <Math/Matrix.h>
#include <Math/Vector.h>
#include <MsgPackUtils.h>
#include <numeric>
#include "Utilities.h"
#include "StandardTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(MatrixVectorAlgebraTest)
	{
		/// <summary>
		/// Creates random instance of Vector class of the given dimension
		/// </summary>
		static Vector VectorFactory(const std::size_t dim = 10)
		{
			return  Vector(dim, -1, 1);
		}

		/// <summary>
		/// Creates random instance of Matrix class of the given dimensions
		/// </summary>
		static Matrix MatrixFactory(const std::size_t row_dim = 10, const std::size_t col_dim = 33)
		{
			return  Matrix(row_dim, col_dim, -1, 1);
		}

	public:

		TEST_METHOD(VectorCopyConstructorTest)
		{
			StandardTestUtils::CopyConstructorTest<Vector>([]() {return VectorFactory();  });
		}

		TEST_METHOD(VectorCopyAssignmentOperatorTest)
		{
			StandardTestUtils::AssignmentOperatorTest<Vector>([]() {return VectorFactory(10); }, []() {return VectorFactory(15); });
		}

		TEST_METHOD(VectorMoveConstructorTest)
		{
			StandardTestUtils::MoveConstructorTest<Vector>([]() {return VectorFactory();  });
		}

		TEST_METHOD(VectorMoveAssignmentTest)
		{
			StandardTestUtils::MoveAssignmentTest<Vector>([]() {return VectorFactory();  });
		}

		TEST_METHOD(MatrixCopyConstructorTest)
		{
			StandardTestUtils::CopyConstructorTest<Matrix>([]() {return MatrixFactory();  });
		}

		TEST_METHOD(MatrixCopyAssignmentOperatorTest)
		{
			StandardTestUtils::AssignmentOperatorTest<Matrix>([]() {return MatrixFactory(10, 33); }, []() {return MatrixFactory(15, 35); });
		}

		TEST_METHOD(MatrixMoveConstructorTest)
		{
			StandardTestUtils::MoveConstructorTest<Matrix>([]() {return MatrixFactory();  });
		}

		TEST_METHOD(MatrixMoveAssignmentTest)
		{
			StandardTestUtils::MoveAssignmentTest<Matrix>([]() {return MatrixFactory();  });
		}

		TEST_METHOD(MatrixVectorMultiplicationTest)
		{
			//Arrange
			const std::size_t row_dim = 10;
			const std::size_t col_dim = 23;
			const auto matrix = Matrix(row_dim, col_dim, -1, 1);
			const auto vector = Vector(col_dim, -1, 1);

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
			const auto matrix = Matrix(row_dim, col_dim, -1, 1);
			const auto vector = Vector(row_dim, -1, 1);

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

		TEST_METHOD(MatrixTransposeTest)
		{
			//Arrange
			const auto matrix = MatrixFactory();
			Assert::IsTrue(matrix.max_abs() > 0, L"Matrix is supposed to be nonzero");
			const auto vector = Vector(matrix.row_dim(), -1, 1);
			Assert::IsTrue(vector.max_abs() > 0, L"Vector is supposed to be nonzero");

			//Act
			const auto matrix_transposed = matrix.transpose();

			//Assert
			const auto diff = (vector * matrix - matrix_transposed * vector).max_abs();
			StandardTestUtils::LogRealAndAssertLessOrEqualTo("Difference", diff, 10 * std::numeric_limits<Real>::epsilon());
		}

		TEST_METHOD(VectorPackingTest)
		{
			StandardTestUtils::PackingTest<Vector>([]() { return VectorFactory(); });
		}

		TEST_METHOD(MatrixPackingTest)
		{
			StandardTestUtils::PackingTest<Matrix>([]() { return MatrixFactory(); });
		}

		TEST_METHOD(SumWithZeroVectorTest)
		{
			const auto dim = 10;
			StandardTestUtils::SumWithZeroElementTest<Vector>([]() { return VectorFactory(dim); }, Vector(dim));
		}

		TEST_METHOD(VectorAdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<Vector>([]() { return VectorFactory(); });
		}

		TEST_METHOD(DifferenceOfEqualVectorsIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<Vector>([]() { return VectorFactory(); });
		}

		TEST_METHOD(VectorAdditionAssocoativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<Vector>([]() { return VectorFactory(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWithRespectToVectorAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<Vector>([]() { return VectorFactory(); });
		}

		TEST_METHOD(VectorMultiplicationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<Vector>([]() { return VectorFactory(); });
		}

		TEST_METHOD(SumWithZeroMatrixTest)
		{
			const auto row_dim = 10;
			const auto col_dim = 23;
			StandardTestUtils::SumWithZeroElementTest<Matrix>([]() { return MatrixFactory(row_dim, col_dim); }, Matrix(row_dim, col_dim));
		}

		TEST_METHOD(MatrixAdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<Matrix>([]() { return MatrixFactory(); });
		}

		TEST_METHOD(DifferenceOfEqualMatricesIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<Matrix>([]() { return MatrixFactory(); });
		}

		TEST_METHOD(MatrixAdditionAssociativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<Matrix>([]() { return MatrixFactory(); });
		}

		TEST_METHOD(DistributivityOfScalarMltiplicationWithRespectToMatrixAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<Matrix>([]() { return MatrixFactory(); });
		}

		TEST_METHOD(MatrixMultiplicationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<Matrix>([]() { return MatrixFactory(); });
		}

		TEST_METHOD(DistributivityOfMatrixAdditionWithRespectToRightVectorMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 10;
			const auto col_dim = 23;
			const auto matrix1 = Matrix(row_dim, col_dim, -1, 1);
			const auto matrix2 = Matrix(row_dim, col_dim, -1, 1);
			const auto vector = Vector(col_dim, -1, 1);
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
			const auto matrix1 = Matrix(row_dim, col_dim, -1, 1);
			const auto matrix2 = Matrix(row_dim, col_dim, -1, 1);
			const auto vector = Vector(row_dim, -1, 1);
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
			const auto matrix = Matrix(row_dim, col_dim, -1, 1);
			const auto vector1 = Vector(col_dim, -1, 1);
			const auto vector2 = Vector(col_dim, -1, 1);
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
			const auto matrix = Matrix(row_dim, col_dim, -1, 1);
			const auto vector1 = Vector(row_dim, -1, 1);
			const auto vector2 = Vector(row_dim, -1, 1);
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
			const auto vec_col = Vector(row_dim, -1, 1);
			const auto vec_row = Vector(col_dim, -1, 1);
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
			const auto vector1 = Vector(dim, -1, 1);
			const auto vector2 = Vector(dim, -1, 1);
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
			const auto matrix = Matrix(row_dim, col_dim, -1, 1);
			const auto mul_vector = Vector(col_dim, -1, 1);
			const auto add_vector = Vector(row_dim, -1, 1);

			//Act
			const auto result1 = matrix * mul_vector + add_vector;
			const auto result2 = matrix.mul_add(mul_vector, add_vector);

			//Assert
			const auto diff = (result1 - result2).max_abs();
			Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + "\n").c_str());
			Assert::IsTrue(diff < 10 * std::numeric_limits<Real>::epsilon(), L"Unexpectedly high difference");
		}

		TEST_METHOD(VectorRandomSelectonMapTest)
		{
			StandardTestUtils::RandomSelectionMapTest<Vector>([](const auto dim) { return VectorFactory(dim); });
		}
	};
}
