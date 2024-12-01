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
#include <numeric>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(CudaVectorTest)
	{
		TEST_METHOD(StandardRandomFillTest)
		{
			//Arrange
			const auto dim = 10000;
			const auto sigma = Utils::get_random(static_cast<Real>(0.1), static_cast<Real>(10));
			auto vector = CudaVector(dim);

			//Act
			vector.standard_random_fill(sigma);

			//Assert
			const auto vector_host = vector.to_stdvector();
			const auto mean = std::accumulate(vector_host.begin(), vector_host.end(), Real(0))/dim;
			const auto sigma_estimated = std::sqrt(std::transform_reduce(vector_host.begin(), vector_host.end(), Real(0), std::plus<Real>(),
				[mean](const auto& x) { return (x - mean) * (x - mean); })/dim);
			const auto sigma_diff = std::abs(sigma - sigma_estimated)/ sigma;

			StandardTestUtils::Log("Mean", mean);
			StandardTestUtils::Log("sigma_diff", sigma_diff);
			Assert::IsTrue(std::abs(mean) < 0.35, L"Unexpectedly high deviation of the \"mean\" value from the reference");
			Assert::IsTrue(std::abs(sigma_diff) < 0.02, L"Unexpectedly high deviation of the \"sigma\" value from the reference");
		}

		/// <summary>
		/// Returns random instance of CudaVector 
		/// </summary>
		static CudaVector CudaVectorFactory(const std::size_t dim = 10)
		{
			return CudaVector(dim, -1, 1);
		}

		TEST_METHOD(CopyConstructorTest)
		{
			StandardTestUtils::CopyConstructorTest<CudaVector>([]() {return CudaVectorFactory(); });
		}

		TEST_METHOD(AssignmentOperatorTest)
		{
			StandardTestUtils::AssignmentOperatorTest<CudaVector>([]() {return CudaVectorFactory(10); }, []() {return CudaVectorFactory(15); });
		}

		TEST_METHOD(MoveConstructorTest)
		{
			StandardTestUtils::MoveConstructorTest<CudaVector>([]() {return CudaVectorFactory(); });
		}

		TEST_METHOD(MoveAssignmentTest)
		{
			StandardTestUtils::MoveAssignmentTest<CudaVector>([]() {return CudaVectorFactory(); });
		}

		TEST_METHOD(PackingTest)
		{
			StandardTestUtils::PackingTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(SumWithZeroVectorTest)
		{
			const auto dim = 10;
			StandardTestUtils::SumWithZeroElementTest<CudaVector>([]() { return CudaVectorFactory(dim); }, CudaVector(dim));
		}

		TEST_METHOD(AdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(DifferenceOfEqualVectorsIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(AdditionAssociativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWithRespectToVectorAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(MultiplicationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(MixedArithmeticTest)
		{
			StandardTestUtils::CudaMixedArithmeticTest<CudaVector>([]() { return CudaVectorFactory(); });
		}

		TEST_METHOD(MaxAbsTest)
		{
			//Arrange
			StandardTestUtils::CudaMaxAbsTest<CudaVector>([]() { return CudaVectorFactory(); });
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
			const auto host_result = cuda_v1.to_host().hadamard_prod(cuda_v2.to_host());
			Assert::IsTrue(cuda_result.to_host() == host_result, L"Actual and reference vectors are not equal");
		}

		TEST_METHOD(HadamardProdAddTest)
		{
			//Arrange
			const auto v0 = CudaVectorFactory();
			const auto v1 = CudaVectorFactory();
			const auto v2 = CudaVectorFactory();

			Assert::IsTrue(v0 != v1 && v2.max_abs() > 0,
				L"Vectors are supposed to be different and nonzero");

			//Act
			auto result = v2;
			result.hadamard_prod_add(v0, v1);

			//Assert
			CudaVector result_expected(v0.size());
			result_expected.hadamard_prod(v0, v1);
			result_expected += v2;

			Assert::IsTrue((result - result_expected).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Too high difference between actual and expected values");
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
			const auto host_result = cuda_v1.to_host().dot_product(cuda_v2.to_host());
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
			const auto host_result = cuda_v1.to_host().sum();
			Assert::IsTrue(std::abs(cuda_result - host_result) < 10 * std::numeric_limits<Real>::epsilon(),
				L"Too high deviation from reference");
		}

		TEST_METHOD(RandomSelectionMapTest)
		{
			StandardTestUtils::RandomSelectionMapTest<CudaVector>([](const auto dim) { return CudaVectorFactory(dim); });
		}

		TEST_METHOD(FillZeroTest)
		{
			// Arrange
			CudaVector vec(100, -1, 1);
			Assert::IsTrue(vec.max_abs() > 0, L"Vector should be non-zero");

			// Act
			vec.fill_zero();

			// Assert
			Assert::IsTrue(vec.max_abs() == 0, L"Vector is supposed to be zero");
		}
	};
}