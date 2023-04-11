//Copyright (c) 2023 Denys Dragunov, dragunovdenis@gmail.com
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
#include "StandardTestUtils.h"
#include "Math/VectorNd.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(VectorNdTest)
	{
		using VectorInTest = VectorNd<Real, 11>;

		/// <summary>
		/// Returns random vector
		/// </summary>
		static VectorInTest get_random_vector()
		{
			return VectorInTest(Utils::get_random_std_vector(11, -1, 1));
		}

		TEST_METHOD(PackingTest)
		{
			//Arrange
			const auto vec = get_random_vector();

			//Act
			const auto msg = MsgPack::pack(vec);
			const auto vec_unpacked = MsgPack::unpack<VectorInTest>(msg);

			//Assert
			Assert::IsTrue(vec == vec_unpacked, L"De-serialized instance is not equal to the original one.");
		}

		TEST_METHOD(SumWithZeroVectorTest)
		{
			StandardTestUtils::SumWithZeroElementTest<VectorInTest>([]() { return get_random_vector(); }, VectorInTest{});
		}

		TEST_METHOD(VectorAdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<VectorInTest>([]() { return get_random_vector(); });
		}

		TEST_METHOD(DifferenceOfEqualVectoreIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<VectorInTest>([]() { return get_random_vector(); });
		}

		TEST_METHOD(VectorAdditionAssociativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<VectorInTest>([]() { return get_random_vector(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWuthRespectToVectorAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<VectorInTest>([]() { return get_random_vector(); });
		}

		TEST_METHOD(VectorMultiplecationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<VectorInTest>([]() { return get_random_vector(); });
		}

		TEST_METHOD(VectorUnaryMinusOperatorTest)
		{
			StandardTestUtils::UnaryMinusOperatorTest<VectorInTest>([]() { return get_random_vector(); });
		}

		TEST_METHOD(VectorConversionToStdVectorAndBackTest)
		{
			//Arrange
			const auto vec = get_random_vector();

			//Act
			VectorInTest vec_restored;
			vec_restored.assign(vec.to_std_vector());

			//Assert
			Assert::IsTrue(vec == vec_restored, L"Vectors are not the same");
		}

		TEST_METHOD(VectorConversionToStringAndBackTest)
		{
			//Arrange
			const auto vec = get_random_vector();

			//Act
			VectorInTest vec_restored;
			auto str = vec.to_string();
			const auto parse_result = VectorInTest::try_parse(str, vec_restored);

			//Assert
			Assert::IsTrue(parse_result, L"Parsing has failed");
			const auto diff = (vec - vec_restored).max_abs();
			Logger::WriteMessage(std::format("Difference between the original and restored vectors: {}\n", diff).c_str());
			Assert::IsTrue(diff < 1e-6, L"Unexpectedly high difference between the original and restored vectors");
		}
	};
}
