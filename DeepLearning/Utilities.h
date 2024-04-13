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
#include <string>
#include <sstream>
#include <vector>
#include <filesystem>

namespace DeepLearning::Utils
{
    /// <summary>
	/// Returns seeded Mersenne Twister generator
	/// </summary>
	inline std::mt19937 get_mt_generator()
    {
        return std::mt19937(std::random_device()());
    }

    /// <summary>
    /// Functionality to fill the given range with uniformly distributed pseudo-random numbers
    /// </summary>
    template< class Iter>
    void fill_with_random_values(Iter start, Iter end, const Real min, const Real max, std::mt19937& gen)
    {
        std::uniform_real_distribution<Real> dist(min, max);
        std::generate(start, end, [&]() { return dist(gen); });
    }

    /// <summary>
    /// Functionality to fill the given range with uniformly distributed pseudo-random numbers
    /// </summary>
    template< class Iter>
    void fill_with_random_values(Iter start, Iter end, const Real min, const Real max)
    {
        thread_local auto gen = get_mt_generator();
        fill_with_random_values(start, end, min, max, gen);
    }

    /// <summary>
    /// Returns std::vector of the given size filled
    /// with uniformly distributed pseudo-random values within the given boundaries
    /// </summary>
    /// <param name="size">Size of the output vector</param>
    /// <param name="min">Lower boundary for the pseudo-random values to populate the collection</param>
    /// <param name="max">Upper boundary for the pseudo-random values to populate the collection</param>
    std::vector<Real> get_random_std_vector(const std::size_t& size, const Real min, const Real max);

    /// <summary>
    /// Returns std::vector of the given size filled
    /// with the values provided by the generator.
    /// </summary>
    std::vector<Real> get_random_std_vector(const int size, const Real min, const Real max, std::mt19937& gen);

    /// <summary>
    /// Returns uniformly distributed random value from the given interval
    /// </summary>
    Real get_random(const Real min, const Real max);

    /// <summary>
    /// Returns uniformly distributed random integer value from the specified interval (ends are included)
    /// </summary>
    int get_random_int(const int min = 0, const int max = std::numeric_limits<int>::max());

    /// <summary>
    /// Functionality to fill the given range with normally distributed pseudo-random numbers
    /// </summary>
    template< class Iter>
    void fill_with_normal_random_values(Iter start, Iter end, const Real mean, const Real std)
    {
        thread_local auto gen = get_mt_generator();
        thread_local std::normal_distribution<Real> dist{ mean, std };

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
    /// Returns 64  bit hash of the given string in hexadecimal representation (16 characters)
    /// </summary>
    std::string get_hash_as_hex_str(const std::string& str);

    /// <summary>
    /// Converts the given hexadecimal string to a double and returns the result
    /// </summary>
    double hex_to_double(const std::string& hex_str);

    /// <summary>
    /// Converts given hexadecimal string to float
    /// </summary>
    template <class T>
    T hex_to_float(const std::string& hex_str)
    {
        return static_cast<T>(hex_to_double(hex_str));
    }

    /// <summary>
    /// Converts the given 64 bit unsigned integer to hexadecimal string
    /// </summary>
    std::string uint64_to_hex(const uint64_t& val);

    /// <summary>
    /// Converts the given double precision value to hexadecimal string and returns the result
    /// </summary>
    std::string double_to_hex(const double& val);

    /// <summary>
    /// Converts given floating point value to its string hexadecimal representation
    /// </summary>
    template <class R>
    std::string float_to_hex(const R& val)
    {
        static_assert(std::is_floating_point_v<R>, "Only for floating point types");

        return double_to_hex(static_cast<double>(val));
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
    /// Extracts and returns a single leftmost (with respect to the given start position)
    /// word from the given string (the given string is updated)
    /// By word we mean a sequence of characters without spaces (whose boundaries are defined by spaces)
    /// which starts at the given `start_id` position
    /// </summary>
    std::string extract_word(std::string& str, const std::size_t& start_id = 0);

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
    /// From the given string extracts a sub-string enclosed within the left-most `open_char` character and the following
    /// `close_char` character so that the sub-string is balanced with respect to the `open_char` and `close_char` characters.
    /// By "balanced" here we mean that for each given character of the sub-string other than `open_char` and `close_char`
    /// the difference between the numbers of `open_char` and 'close_char' characters to the left is nonnegative
    /// and equal to the difference between numbers of `close_char` and 'open_char' characters to the right.
    /// The sub-string is returned and erased from the input string (including the "defining" `open_char`-`close_char` pair).
    /// Throws exception if either of `open_char`, 'cloase_char' characters is present in the input string but
    /// a "balanced" sub-string can't be extracted;
    /// </summary>
    std::string extract_balanced_sub_string(std::string& str, const char open_char = '{',
        const char close_char = '}', const bool keep_enclosure = false);

    /// <summary>
    /// A "try" pattern version of the corresponding method above that does not throw
    /// exceptions in case it is not possible to extract the sub-string
    /// </summary>
    /// <param name="str">Input string that gets modified in case method succeeds</param>
    /// <param name="out">The extracted sub-string in case method succeeds or "garbage" otherwise</param>
    /// <param name="open_char">The "open" character</param>
    /// <param name="close_char">The "close" character</param>
    bool try_extract_balanced_sub_string(std::string& str, std::string& out, const char open_char = '{', const char close_char = '}');

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

    /// <summary
    /// Returns string representation of the given vector
    /// </summary>
    template <class T>
    std::string vector_to_str(const std::vector<T>& vec)
    {
        if (vec.empty())
            return "{}";

        std::stringstream ss;
        ss << "{";

        for (auto item_id = 0ull; item_id < vec.size(); ++item_id)
        {
            ss << vec[item_id];
            if (item_id != vec.size() - 1)
                ss << ", ";
            else
                ss << "}";
        }

        return ss.str();
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
    /// Parses as many vectors of type "V" from the given string as possible and returns them as a collection
    /// </summary>
    template <class V>
    std::vector<V> extract_vectors(const std::string& str);

    /// <summary>
    /// Parses given string and extracts a vector of type V from it. Throws an exception if parsing fails.
    /// </summary>
    template <class V>
    V extract_vector(const std::string& str);

    /// <summary>
    /// Transforms given collection to string
    /// </summary>
    /// <param name="collection">The collection</param>
    /// <param name="delim">Delimiter used to separate string representation of the collection items</param>
    template <class V>
    std::string to_string(const std::vector<V>& collection, const char delim = '\n');

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

    /// <summary>
    /// Converts given floating point value to an exact string representation,
    /// which is a combination of a human-readable decimal representation (might be not exact)
    /// and an exact hexadecimal representation
    /// </summary>
    template <class R>
    std::string float_to_str_exact(const R& val)
    {
        static_assert(std::is_floating_point_v<R>, "Only for floating point types");
        //A combination of human-readable and exact representation of the number
        return Utils::to_string(val) + "$" + float_to_hex(val);
    }

    /// <summary>
    /// Parses given string to a floating point value, can process the "exact" representation produced by `float_to_str_exact`
    /// </summary>
    template <class R>
    R str_to_float(const std::string& str)
    {
        static_assert(std::is_floating_point_v<R>, "Only for floating point types");

        const auto parts = split_by_char(str, '$');

        if (parts.size() > 2)
            throw std::exception("Invalid input");

        if (parts.size() == 2)
            return hex_to_float<R>(parts[1]);

        if constexpr (std::is_same_v<R, double>)
			return std::stod(parts[0]);
        else if constexpr (std::is_same_v<R, float>)
            return std::stof(parts[0]);

        throw std::exception("Unexpected input type");
    }

    /// <summary>
    /// Returns a globally unique identifier string
    /// </summary>
    std::string create_guid_string();

    /// <summary>
    /// Converts given amount of seconds into dd:hh:mm:ss string representation
    /// </summary>
    std::string seconds_to_dd_hh_mm_ss_string(const long long& time_sec);

    /// <summary>
    /// Converts given amount of milliseconds into dd:hh:mm:ss string representation
    /// </summary>
    std::string milliseconds_to_dd_hh_mm_ss_string(const long long& time_msec);

    /// <summary>
    /// Returns time-string formatted as (an example) "Wed Jun 30 21:49:08 1993".
    /// </summary>
    std::string format_date_time(const std::chrono::system_clock::time_point& time_pt);

    /// <summary>
    /// Returns formatted duration between the start and stop time points.
    /// </summary>
    std::string get_elapsed_time_formatted(const std::chrono::system_clock::time_point& start_pt,
        const std::chrono::system_clock::time_point& stop_pt);

    /// <summary>
    /// Return string with formatted elapsed time and updates given time-point with new measurement.
    /// </summary>
    /// <param name="start_pt">Start time point.</param>
    std::string get_elapsed_time_formatted(std::chrono::system_clock::time_point& start_pt);

    /// <summary>
    /// Returns the argument squared
    /// </summary>
    template <class T>
    T sqr(const T& x)
    {
        return x * x;
    }
}
