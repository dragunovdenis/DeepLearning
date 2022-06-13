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
#include <Utilities.h>
#include "StandardTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(CudaVectorTest)
	{
		/// <summary>
		/// Returns random instance of CudaVector 
		/// </summary>
		static CudaVector CudaVectorFactory(const std::size_t dim = 10)
		{
			return CudaVector(dim, -1, 1);
		}

		TEST_METHOD(CudaVectorCopyConstructorTest)
		{
			StandardTestUtils::CopyConstructorTest<CudaVector>([]() {return CudaVectorFactory(); });
		}

		TEST_METHOD(CudaVectorAssignmentOperatorTest)
		{
			StandardTestUtils::AssignmentOperatorTest<CudaVector>([]() {return CudaVectorFactory(10); }, []() {return CudaVectorFactory(15); });
		}

		TEST_METHOD(CudaVectorMoveConstructorTest)
		{
			StandardTestUtils::MoveConstructorTest<CudaVector>([]() {return CudaVectorFactory(); });
		}

		TEST_METHOD(CudaVectorPackingTest)
		{
			StandardTestUtils::PackingTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(SumWithZeroVectorTest)
		{
			const auto dim = 10;
			StandardTestUtils::SumWithZeroElementTest<CudaVector>([]() { return CudaVectorFactory(dim); }, CudaVector(dim));
		}

		TEST_METHOD(VectorAdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(DifferenceOfEqualVectorsIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(VectorAdditionAssocoativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWithRespectToVectorAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(VectorMultiplicationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		template <class T>
		T mixed_arithmetic_function(const T& v1, const T& v2, const T& v3)
		{
			return (0.5 * v1 + v1 * 3.4 - 5.1 * v3) * 0.75;
		}

		TEST_METHOD(MixedArithmeticTest)
		{
			//Arrange
			const auto cuda_v1 = CudaVectorFactory();
			const auto cuda_v2 = CudaVectorFactory();
			const auto cuda_v3 = CudaVectorFactory();

			Assert::IsTrue(cuda_v1 != cuda_v2 &&
				cuda_v1 != cuda_v3 && cuda_v2 != cuda_v3, L"Vectors are supposed to be different");

			//Act
			const auto cuda_result = mixed_arithmetic_function(cuda_v1, cuda_v2, cuda_v3);

			//Assert
			const auto host_result = mixed_arithmetic_function(
				cuda_v1.to_host_vector(), cuda_v2.to_host_vector(), cuda_v3.to_host_vector());

			Assert::IsTrue((cuda_result.to_host_vector() - host_result).max_abs() < 
				10 * std::numeric_limits<Real>::epsilon(),
				L"Too high deviation from reference");
		}

		TEST_METHOD(MaxAbsTest)
		{
			//Arrange
			const auto cuda_v1 = CudaVectorFactory();

			//Act
			const auto max_abs = cuda_v1.max_abs();

			//Assert
			Assert::IsTrue(max_abs == cuda_v1.to_host_vector().max_abs(), L"Actual and expected values are not equal");
			Assert::IsTrue(max_abs > 0, L"Vector is supposed to be nonzero");
		}

		TEST_METHOD(HadamardProdTest)
		{
			//Arrange
			const auto cuda_v1 = CudaVectorFactory();
			const auto cuda_v2 = CudaVectorFactory();

			Assert::IsTrue(cuda_v1 != cuda_v2, L"Vectors are supposed to be different");

			//Act
			const auto cuda_result = cuda_v1.hadamard_prod(cuda_v2);

			//Assert
			const auto host_result = cuda_v1.to_host_vector().hadamard_prod(cuda_v2.to_host_vector());
			Assert::IsTrue(cuda_result.to_host_vector() == host_result, L"Actual and reference vectors are not equal");
		}

		TEST_METHOD(DotProdTest)
		{
			//Arrange
			const auto cuda_v1 = CudaVectorFactory();
			const auto cuda_v2 = CudaVectorFactory();

			Assert::IsTrue(cuda_v1 != cuda_v2, L"Vectors are supposed to be different");

			//Act
			const auto cuda_result = cuda_v1.dot_product(cuda_v2);

			//Assert
			const auto host_result = cuda_v1.to_host_vector().dot_product(cuda_v2.to_host_vector());
			Assert::IsTrue(std::abs(cuda_result - host_result) < 10 * std::numeric_limits<Real>::epsilon(),
				L"Too high deviation from reference");
		}

		TEST_METHOD(ElementSumTest)
		{
			//Arrange
			const auto cuda_v1 = CudaVectorFactory();

			Assert::IsTrue(cuda_v1.max_abs() > 0, L"Vectors is supposed to be nonzero");

			//Act
			const auto cuda_result = cuda_v1.sum();

			//Assert
			const auto host_result = cuda_v1.to_host_vector().sum();
			Assert::IsTrue(std::abs(cuda_result - host_result) < 10 * std::numeric_limits<Real>::epsilon(),
				L"Too high deviation from reference");
		}
	};
}