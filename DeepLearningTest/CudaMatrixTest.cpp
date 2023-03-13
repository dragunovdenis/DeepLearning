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
#include <Math/CudaVector.cuh>
#include <Math/CudaMatrix.cuh>
#include <Utilities.h>
#include "StandardTestUtils.h"
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(CudaMatrixTest)
	{
		/// <summary>
		/// Returns random instance of CudaMatrix 
		/// </summary>
		static CudaMatrix CudaMatrixFactory(const std::size_t row_dim = 10, const std::size_t col_dim = 13)
		{
			return CudaMatrix(row_dim, col_dim, -1, 1);
		}

		TEST_METHOD(CopyConstructorTest)
		{
			StandardTestUtils::CopyConstructorTest<CudaMatrix>([]() {return CudaMatrixFactory(); });
		}

		TEST_METHOD(AssignmentOperatorTest)
		{
			StandardTestUtils::AssignmentOperatorTest<CudaMatrix>([]() {return CudaMatrixFactory(10, 13); }, []() {return CudaMatrixFactory(11, 25); });
		}

		TEST_METHOD(MoveConstructorTest)
		{
			StandardTestUtils::MoveConstructorTest<CudaMatrix>([]() {return CudaMatrixFactory(); });
		}

		TEST_METHOD(MoveAssignmentTest)
		{
			StandardTestUtils::MoveAssignmentTest<CudaMatrix>([]() {return CudaMatrixFactory(); });
		}

		TEST_METHOD(PackingTest)
		{
			StandardTestUtils::PackingTest<CudaMatrix>([]() { return CudaMatrixFactory(); });
		}

		TEST_METHOD(SumWithZeroVectorTest)
		{
			const auto row_dim = 10;
			const auto col_dim = 13;
			StandardTestUtils::SumWithZeroElementTest<CudaMatrix>([]() { return CudaMatrixFactory(row_dim, col_dim); }, CudaMatrix(row_dim, col_dim));
		}

		TEST_METHOD(AdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<CudaMatrix>([]() { return CudaMatrixFactory(); });
		}

		TEST_METHOD(DifferenceOfEqualVectorsIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<CudaMatrix>([]() { return CudaMatrixFactory(); });
		}

		TEST_METHOD(AdditionAssociativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<CudaMatrix>([]() { return CudaMatrixFactory(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWithRespectToVectorAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<CudaMatrix>([]() { return CudaMatrixFactory(); });
		}

		TEST_METHOD(MultiplicationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<CudaMatrix>([]() { return CudaMatrixFactory(); });
		}

		TEST_METHOD(MixedArithmeticTest)
		{
			StandardTestUtils::CudaMixedArithmeticTest<CudaMatrix>([]() { return CudaMatrixFactory(); });
		}

		TEST_METHOD(MatrixVectorMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 100;
			const auto col_dim = 1000;
			const auto matr = CudaMatrixFactory(row_dim, col_dim);
			const auto vect = CudaVector(col_dim, -1, 1);

			Assert::IsTrue(matr.max_abs() > 0, L"Matrix is supposed to be non-zero");
			Assert::IsTrue(vect.max_abs() > 0, L"Vector is supposed to be non-zero");

			//Act
			const auto result = matr * vect;

			//Assert
			const auto host_result = matr.to_host() * vect.to_host();
			const auto diff = (result.to_host() - host_result).max_abs();
			Logger::WriteMessage((std::string("Diff = ") + Utils::to_string(diff) + "\n").c_str());
			Assert::IsTrue(diff < 500 * std::numeric_limits<Real>::epsilon(), L"too high deviation from reference");
		}

		TEST_METHOD(MatrixVectorMulAddTest)
		{
			//Arrange
			const auto row_dim = 100;
			const auto col_dim = 1000;
			const auto matr = CudaMatrixFactory(row_dim, col_dim);
			const auto vect_mul = CudaVector(col_dim, -1, 1);
			const auto vect_add = CudaVector(row_dim, -1, 1);

			Assert::IsTrue(matr.max_abs() > 0, L"Matrix is supposed to be non-zero");
			Assert::IsTrue(vect_mul.max_abs() > 0 && vect_add.max_abs() > 0, L"Vectors are supposed to be non-zero");

			//Act
			const auto result = matr.mul_add(vect_mul, vect_add);;

			//Assert
			const auto host_result = matr.to_host().mul_add(vect_mul.to_host(), vect_add.to_host());
			const auto diff = (result.to_host() - host_result).max_abs();
			Logger::WriteMessage((std::string("Diff = ") + Utils::to_string(diff) + "\n").c_str());
			Assert::IsTrue(diff < 600 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference");
		}

		TEST_METHOD(VectorMatrixMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 100;
			const auto col_dim = 1000;
			const auto matr = CudaMatrixFactory(row_dim, col_dim);
			const auto vect = CudaVector(row_dim, -1, 1);

			Assert::IsTrue(matr.max_abs() > 0, L"Matrix is supposed to be non-zero");
			Assert::IsTrue(vect.max_abs() > 0, L"Vector is supposed to be non-zero");

			//Act
			const auto result = vect * matr;

			//Assert
			const auto host_result = vect.to_host() * matr.to_host();
			const auto diff = (result.to_host() - host_result).max_abs();
			Logger::WriteMessage((std::string("Diff = ") + Utils::to_string(diff) + "\n").c_str());
			Assert::IsTrue(diff < 50 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference");
		}

		TEST_METHOD(VectorColByVectorRowMultiplicationTest)
		{
			//Arrange
			const auto row_dim = 110;
			const auto col_dim = 135;
			const auto vect_col = CudaVector(row_dim, -1, 1);
			const auto vect_row = CudaVector(col_dim, -1, 1);

			Assert::IsTrue(vect_col.max_abs() > 0 && vect_row.max_abs() > 0, L"Input vectors are supposed to be non-zero");

			//Act
			const auto result = vector_col_times_vector_row(vect_col, vect_row);

			//Assert
			const auto host_result = vector_col_times_vector_row(vect_col.to_host(), vect_row.to_host());
			const auto diff = (result.to_host() - host_result).max_abs();
			Logger::WriteMessage((std::string("Diff = ") + Utils::to_string(diff) + "\n").c_str());
			Assert::IsTrue(diff < 10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference");
		}

		TEST_METHOD(TransposeTest)
		{
			//Arrange
			const auto matrix = CudaMatrixFactory(3, 5);
			Assert::IsTrue(matrix.max_abs() > 0, L"Matrix is supposed to be nonzero");

			//Act
			const auto matrix_transposed = matrix.transpose();

			//Assert
			Assert::IsTrue(matrix_transposed.to_host() == matrix.to_host().transpose(), L"Transposed matrix deviates from reference");
		}
	};
}
