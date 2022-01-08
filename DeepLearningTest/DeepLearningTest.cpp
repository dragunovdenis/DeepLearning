
#include "CppUnitTest.h"
#include <Math/DenseMatrix.h>
#include <Math/DenseVector.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(DenseMatrixAndVectorTest)
	{
	public:
		
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
				Logger::WriteMessage((std::string("Difference = ") + std::to_string(diff) + "\n").c_str());
				Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(), L"Unexpectedly high difference");
			}
		}

		TEST_METHOD(VectorMatrixMultiplicationTest)
		{
			//Arrange
			const std::size_t row_dim = 43;
			const std::size_t col_dim = 14;
			auto matrix = DenseMatrix(row_dim, col_dim, -1, 1);
			auto vector = DenseVector(row_dim, -1, 1);

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
				Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(), L"Unexpectedly high difference");
			}
		}
	};
}
