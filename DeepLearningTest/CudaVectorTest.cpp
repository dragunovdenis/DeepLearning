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
	};
}