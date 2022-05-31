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

#pragma once

#include <random>
#include <algorithm>
#include <ios>
#include "defs.h"
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <filesystem>

namespace DeepLearning::Utils
{
    /// <summary>
    /// Functionality to fill the given range with uniformly distributed pseudo-random numbers
    /// </summary>
    template< class Iter>
    void fill_with_random_values(Iter start, Iter end, const Real min, const Real max)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

        const auto normalization_factor = (max - min) / std::numeric_limits<int>::max();
        std::generate(start, end, [&]() { return normalization_factor * dist(gen) + min; });
    }

    /// <summary>
    /// Returns uniformly distributed random value from the given interval
    /// </summary>
    Real get_random(const Real min, const Real max);

    /// <summary>
    /// Returns uniformly distributed random integer value from the specified interval (ends are included)
    /// </summary>
    int get_random_int(const int min, const int max);

    /// <summary>
    /// Functionality to fill the given range with normally distributed pseudo-random numbers
    /// </summary>
    template< class Iter>
    void fill_with_normal_random_values(Iter start, Iter end, const Real mean, const Real std)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<Real> dist{ mean, std };

        std::generate(start, end, [&]() { return dist(gen); });
    }

    /// <summary>
    /// Converts given value to string with the given number of digits
    /// </summary>
    template <typename R>
    std::string to_string(const R a_value, const int n = std::numeric_limits<R>::digits10)
    {
        std::ostringstream out;
        out.precision(n);
        out << std::fixed << a_value;
        return out.str();
    }

    /// <summary>
    /// An analogous of the Python's nan_to_num() function
    /// </summary>
    template <class R>
    R nan_to_num(const R& val) {
        if (std::isinf(val)) {
            if (val < R(0))
                return -std::numeric_limits<R>::max();
            else
                return std::numeric_limits<R>::max();
        }
        else if (std::isnan(val)) {
            return R(0);
        }

        return val;
    }

    /// <summary>
    /// Substitutes "," and ";" characters with spaces
    /// </summary>
    std::string normalize_separator_characters(const std::string& str);

    /// <summary>
    /// Parses given string, which is supposed to contain comma, semicolon or space separated values of type `R`
    /// Does not parse past the first illegal character
    /// </summary>
    template <class R>
    std::vector<R> parse_scalars(const std::string& str)
    {
        const auto str_clean = normalize_separator_characters(str);
        std::istringstream str_clean_stream(str_clean);

        std::vector<R> result;
        R temp;
        while (str_clean_stream >> temp)
            result.push_back(temp);

        return result;
    }

    /// <summary>
    /// Extracts and returns a single leftmost word from the given string (the given string is updated)
    /// By word we mean a sequence of characters without spaces (whose boundaries are defined by spaces)
    /// </summary>
    std::string extract_word(std::string& str);

    /// <summary>
    /// From the given string extracts the leftmost sub-string of the form "{...}" (the input string gets updated)
    /// Returns what is enclosed in curly brackets of the extracted sub-string
    /// Throws an exception if no valid sub-string of the form "{...}" is found
    /// </summary>
    std::string extract_vector_sub_string(std::string& str);

    /// <summary>
    /// A "try" pattern version of the corresponding method above that does not throw
    /// exceptions in case it is not possible to extract the sub-string
    /// </summary>
    /// <param name="str">Input string that gets modified in case method succeeds</param>
    /// <param name="out">The extracted sub-string in case method succeeds or "garbage" otherwise</param>
    bool try_extract_vector_sub_string(std::string& str, std::string& out);

    /// <summary>
    /// From the given string extracts and parse the leftmost sub-string of the form "{...}" (the input string gets updated)
    /// The sub-string in the brackets is supposed to contain comma, semicolon or space separated strings that can be parsed
    /// into variables of type `R`; Returns vector of parsed values
    /// </summary>
    template <class R>
    std::vector<R> parse_vector(std::string& str)
    {
        return parse_scalars<R>(extract_vector_sub_string(str));
    }

    /// <summary>
    /// Removes leading trailing and extra spaces from the input string and returns the result
    /// </summary>
    std::string remove_leading_trailing_extra_spaces(const std::string& str);

    /// <summary>
    /// Converts the given string to upper case and returns the result
    /// </summary>
    std::string to_upper_case(const std::string& str);

    /// <summary>
    /// Returns a "normalized" version of the given string, which is a string
    /// with all "," and ";" characters replaced with spaces,
    /// containing nor leading, trailing or extra spaces, with all the letters in the upper case
    /// </summary>
    std::string normalize_string(const std::string& str);

    /// <summary>
    /// Tries to extract vector of type "V" from the given string
    /// The corresponding part of the given string gets removed in case function succeeds
    /// </summary>
    /// <param name="str">given input string. It will be modified in case call succeeds</param>
    /// <param name="out">Parsed vector (in case call succeeds) or "garbage" (otherwise)</param>
    /// <returns></returns>
    template <class V>
    bool try_extract_vector(std::string& str, V& out);

    /// <summary>
    /// Splits given string with respect to the given delimiter
    /// </summary>
    /// <param name="str">String to split</param>
    /// <param name="delim">Delimiter</param>
    /// <returns></returns>
    std::vector<std::string> split_by_char(const std::string& str, const char delim);

    /// <summary>
    /// Reads all the text from the given text file and returns it as a string
    /// </summary>
    /// <param name="file_name">Full path to a text file to read</param>
    std::string read_all_text(const std::filesystem::path& file_name);
}
