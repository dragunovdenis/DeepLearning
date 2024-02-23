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

#include <numeric>
#include "CppUnitTest.h"
#include "RandomGenerator.h"
#include "StandardTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearningTest::StandardTestUtils;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(RandomGeneratorTest)
	{
		TEST_METHOD(DistributionTest)
		{
			// Arrange
			RandomGenerator generator;
			constexpr auto range = 100;
			std::vector<int> values_counter(range);
			constexpr auto samples_per_value = 1000;
			constexpr auto samples_total = range * samples_per_value;
			auto samples_counter = 0;

			while (samples_total > samples_counter++)
			{
				const auto value = generator.get_int(0, range);
				Assert::IsTrue(value >= 0 && value < range, L"Requested range exceeded");
				values_counter[value]++;
			}

			const auto min_value_count = *std::ranges::min_element(values_counter);
			const auto max_value_count = *std::ranges::max_element(values_counter);

			LogReal("Minimal number of registered samples", min_value_count);
			LogReal("Maximal number of registered samples", max_value_count);

			Assert::IsTrue(min_value_count >= samples_per_value * 0.85, L"Too few values registered.");
			Assert::IsTrue(max_value_count <= samples_per_value * 1.15, L"Too many values registered.");
		}
	};
}
