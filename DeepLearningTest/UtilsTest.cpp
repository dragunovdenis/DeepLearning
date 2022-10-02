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
#include "Utilities.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(UtilsTest)
	{
		TEST_METHOD(ParseDoubleScalarsTest)
		{
			//Arrange
			std::string string_to_parse = "  ,;-1.2, 4.6, 7;,;5 6.2";
			std::vector<double> expected_result = { -1.2, 4.6, 7.0, 5.0, 6.2 };

			//Act
			const auto result = Utils::parse_scalars<double>(string_to_parse);

			//Assert
			Assert::IsTrue(expected_result == result, L"Actual parsed values do not coincide with the expected value");
		}

		TEST_METHOD(ParseDoubleScalarsInvalidStringTest)
		{
			//Arrange
			std::string string_to_parse = "  ,;-1.2, 4.6, something 7;,;5 6.2";
			std::vector<double> expected_result = { -1.2, 4.6 };

			//Act
			const auto result = Utils::parse_scalars<double>(string_to_parse);

			//Assert
			Assert::IsTrue(expected_result == result, L"Actual parsed values do not coincide with the expected value");
		}

		TEST_METHOD(ParseIntScalarsTest)
		{
			//Arrange
			std::string string_to_parse = "  ,;-12, 46, 7;,;5 62";
			std::vector<int> expected_result = { -12, 46, 7, 5, 62 };

			//Act
			const auto result = Utils::parse_scalars<int>(string_to_parse);

			//Assert
			Assert::IsTrue(expected_result == result, L"Actual parsed values do not coincide with the expected value");
		}

		TEST_METHOD(ParseIntScalarsFromStringWithDoublesTest)
		{
			//Arrange
			std::string string_to_parse = "  ,;-1.2, 4.6, 7;,;5 6.2";
			std::vector<int> expected_result = { -1 };

			//Act
			const auto result = Utils::parse_scalars<int>(string_to_parse);

			//Assert
			Assert::IsTrue(expected_result == result, L"Actual parsed values do not coincide with the expected value");
		}

		TEST_METHOD(ExtractVectorStringTest)
		{
			//Arrange
			std::string string_to_parse = "  ,;-1.2,something{4.6, 7} ; else,;{5 6.2}";
			const std::string expected_string_after_extraction = "  ,;-1.2,something ; else,;{5 6.2}";
			const std::string expected_extracted_string = "4.6, 7";

			//Act
			const auto extracted_string = Utils::extract_vector_sub_string(string_to_parse);

			//Assert
			Assert::IsTrue(expected_string_after_extraction == string_to_parse, L"Unexpected value of the input string");
			Assert::IsTrue(extracted_string == expected_extracted_string, L"Unexpected value of the extracted string");
		}

		TEST_METHOD(ParseVectorTest)
		{
			//Arrange
			std::string string_to_parse = "  ,;-1.2,something{4.6, 7} ; else,;{5 6.2}";
			const std::string expected_string_after_extraction = "  ,;-1.2,something ; else,;{5 6.2}";
			const std::vector<double> expected_extracted_string = { 4.6, 7.0 };

			//Act
			const auto extracted_string = Utils::parse_vector<double>(string_to_parse);

			//Assert
			Assert::IsTrue(expected_string_after_extraction == string_to_parse, L"Unexpected value of the input string");
			Assert::IsTrue(extracted_string == expected_extracted_string, L"Unexpected value of the extracted string");
		}

		TEST_METHOD(RemoveLeadingTrailingExtraSpacesTest)
		{
			//Arrange
			const std::string input_string = "     some   text 345 1  44  and    numbers    ";
			const std::string expected_result = "some text 345 1 44 and numbers";

			//Act
			const auto result = Utils::remove_leading_trailing_extra_spaces(input_string);

			//Assert
			Assert::IsTrue(result == expected_result, L"Unexpected result");
		}

		TEST_METHOD(ToUpperCaseTest)
		{
			//Arrange
			const std::string input_string = "c 23 aBcdEFg";
			const std::string expected_result = "C 23 ABCDEFG";

			//Act
			const auto result = Utils::to_upper_case(input_string);

			//Assert
			Assert::IsTrue(result == expected_result, L"Unexpected result");
		}

		TEST_METHOD(ExtractWordTest)
		{
			//Arrange
			std::string input_string = "    word1    some   other  word   ";
			const std::string input_string_after_extraction_expected = "    some   other  word   ";
			const std::string expected_result = "word1";

			//Act
			const auto result = Utils::extract_word(input_string);

			//Assert
			Assert::IsTrue(input_string == input_string_after_extraction_expected, L"Unexpected value of the input string after extraction");
			Assert::IsTrue(expected_result == result, L"Unexpected extraction result");
		}

		TEST_METHOD(ExtractWordInTheMiddleTest)
		{
			//Arrange
			std::string input_string = "    word1    some   other  word   ";
			const std::string input_string_after_extraction_expected = "    word1   other  word   ";
			const std::string expected_result = "some";

			//Act
			const auto result = Utils::extract_word(input_string, 9);

			//Assert
			Assert::IsTrue(input_string == input_string_after_extraction_expected, L"Unexpected value of the input string after extraction");
			Assert::IsTrue(expected_result == result, L"Unexpected extraction result");
		}

		TEST_METHOD(ExtractLastWordTest)
		{
			//Arrange
			std::string input_string = "    word1";
			const std::string input_string_after_extraction_expected = "";
			const std::string expected_result = "word1";

			//Act
			const auto result = Utils::extract_word(input_string);

			//Assert
			Assert::IsTrue(input_string == input_string_after_extraction_expected, L"Unexpected value of the input string after extraction");
			Assert::IsTrue(expected_result == result, L"Unexpected extraction result");
		}

		TEST_METHOD(ExtractWordWhereThereIsNoWordsTest)
		{
			//Arrange
			std::string input_string = "    ";
			const std::string input_string_after_extraction_expected = "";
			const std::string expected_result = "";

			//Act
			const auto result = Utils::extract_word(input_string);

			//Assert
			Assert::IsTrue(input_string == input_string_after_extraction_expected, L"Unexpected value of the input string after extraction");
			Assert::IsTrue(expected_result == result, L"Unexpected extraction result");
		}

		TEST_METHOD(DoubleToHexAndBackTest)
		{
			//Arrange
			const auto scalar = static_cast<double>(Utils::get_random(0, 1));

			//Act
			const auto hex_string = Utils::float_to_hex(scalar);
			const auto scalar_decoded = Utils::hex_to_float<double>(hex_string);

			//Assert
			Assert::IsTrue(scalar == scalar_decoded, L"Encoding-decoding failed");
		}

		TEST_METHOD(FloatToHexAndBackTest)
		{
			//Arrange
			const auto scalar = static_cast<float>(Utils::get_random(0, 1));

			//Act
			const auto hex_string = Utils::float_to_hex(scalar);
			const auto scalar_decoded = Utils::hex_to_float<float>(hex_string);

			//Assert
			Assert::IsTrue(scalar == scalar_decoded, L"Encoding-decoding failed");
		}

		TEST_METHOD(FloatToStrExactAndBackTest)
		{
			//Arrange
			const auto scalar = Utils::get_random(0, 1);

			//Act
			const auto string_exact = Utils::float_to_str_exact(scalar);
			const auto scalar_decoded = Utils::str_to_float<Real>(string_exact);

			//Assert
			Assert::IsTrue(scalar == scalar_decoded, L"Encoding-decoding failed");
			const auto parts = Utils::split_by_char(string_exact, '$');
			Assert::IsTrue(parts.size() == 2, L"Unexpected number of parts");

			const auto diff = std::abs(Utils::str_to_float<Real>(parts[0]) - Utils::hex_to_float<Real>(parts[1]));

			Assert::IsTrue(diff < 100 * std::numeric_limits<Real>::epsilon(),
				L"Too big difference between the two representations of the same floating-point number");
		}
	};
}