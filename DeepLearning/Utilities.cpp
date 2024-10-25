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

#include "Utilities.h"
#include "Math/LinAlg2d.h"
#include "Math/LinAlg3d.h"
#include <regex>
#include <fstream>
#include <OleCtl.h>

namespace DeepLearning::Utils
{
    Real get_random(const Real min, const Real max)
    {
        thread_local auto gen = get_mt_generator();
        std::uniform_real_distribution<Real> dist(min, max);
        return dist(gen);
    }

    std::vector<Real> get_random_std_vector(const std::size_t& size, const Real min, const Real max)
    {
        std::vector<Real> result(size);
        fill_with_random_values(result.begin(), result.end(), min, max);
        return result;
    }

    std::vector<Real> get_random_std_vector(const int size, const Real min, const Real max, std::mt19937& gen)
    {
        std::vector<Real> result(size);
        fill_with_random_values(result.begin(), result.end(), min, max, gen);
        return result;
    }

    int get_random_int(const int min, const int max)
    {
        thread_local auto gen = get_mt_generator();
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }

    std::string extract_word(std::string& str, const std::size_t& start_id)
    {
        const auto word_start_pos = str.find_first_not_of(' ', start_id);

        if (word_start_pos == std::string::npos) // nothing to extract
        {
            str.erase(start_id, str.size() - start_id);
            return std::string("");
        }

        auto word_end_pos = str.find_first_of(' ', word_start_pos);

        if (word_end_pos == std::string::npos)
            word_end_pos = str.length();

        const auto result = str.substr(word_start_pos, word_end_pos - word_start_pos);

        str.erase(start_id, word_end_pos - start_id);

        return result;
    }

    std::string extract_vector_sub_string(std::string& str)
    {
        const auto vector_end_pos = str.find("}");

        if (vector_end_pos == std::string::npos)
            throw std::exception("Closing bracket not found");

        const auto vector_start_pos = str.find("{");

        if (vector_start_pos == std::string::npos || vector_start_pos > vector_end_pos)
            throw std::exception("Opening bracket not found");

        const auto result = str.substr(vector_start_pos + 1, vector_end_pos - vector_start_pos - 1);
        str.erase(vector_start_pos, vector_end_pos - vector_start_pos + 1);

        return result;
    }

    std::string normalize_separator_characters(const std::string& str)
    {
        std::string result;
        std::replace_copy_if(str.begin(), str.end(), std::back_inserter<std::string>(result),
            [](const auto x) {
                return x == ',' || x == ';';
            }, ' ');

        return result;
    }

    std::string get_hash_as_hex_str(const std::string& str)
    {
        return uint64_to_hex(std::hash<std::string>()(str));
    }

    double hex_to_double(const std::string& hex_str) {
        if (hex_str.size() != 16)
            throw std::exception("Invalid input");

        uint64_t i = std::stoll(hex_str, nullptr, 16);
        double d;
        std::memcpy(&d, &i, sizeof(double));
        return d;
    }

    std::string uint64_to_hex(const uint64_t& val)
    {
        constexpr auto buf_size = sizeof(double) * 2 + 1;
        char buf[buf_size];
        if (snprintf(buf, sizeof(buf), "%016llx", val) != sizeof(double) * 2)
            throw std::string("Conversion failed");

        buf[buf_size - 1] = 0;

        return std::string{ buf };
    }

    std::string double_to_hex(const double& val) {
        uint64_t i;
        std::memcpy(&i, &val, sizeof(double));
        return uint64_to_hex(i);
    }

    bool try_extract_vector_sub_string(std::string& str, std::string& out)
    {
        try
        {
            out = extract_vector_sub_string(str);
        }
        catch (const std::exception&)
        {
            return false;
        }

        return true;
    }

    std::string extract_balanced_sub_string(std::string& str, const char open_char, const char close_char, const bool keep_enclosure)
    {
        const auto start_pos = str.find(open_char);
        const auto first_close_char_pos = str.find(close_char);

        if (start_pos == std::string::npos && first_close_char_pos == std::string::npos)
            return "";

        if (first_close_char_pos == std::string::npos || first_close_char_pos < start_pos)
            throw std::exception("Can't extract a balanced sub-string");

        auto opened_cnt = 1;
        auto pos = start_pos;

        while (pos < str.size() && opened_cnt > 0)
        {
            ++pos;
            if (str[pos] == open_char)
                ++opened_cnt;
            else if (str[pos] == close_char)
                --opened_cnt;
        }

        if (opened_cnt != 0)
          throw std::exception("Can't extract a balanced sub-string");

        auto result = keep_enclosure ? str.substr(start_pos, pos - start_pos + 1) :
            str.substr(start_pos + 1, pos - start_pos - 1);

        str.erase(start_pos, pos - start_pos + 1);

        return result;
    }

    bool try_extract_balanced_sub_string(std::string& str, std::string& out, const char open_char, const char close_char)
    {
        try
        {
            out = extract_balanced_sub_string(str, open_char, close_char);
        }
        catch (const std::exception&)
        {
            return false;
        }

        return true;
    }

    std::string remove_leading_trailing_extra_spaces(const std::string& str)
    {
        return std::regex_replace(str, std::regex("^ +| +$|( ) +"), "$1");
    }

    std::string to_upper_case(const std::string& str)
    {
        auto result = str;
        std::ranges::transform(str, result.begin(), ::toupper);

        return result;
    }

    std::string normalize_string(const std::string& str)
    {
        return to_upper_case(Utils::remove_leading_trailing_extra_spaces(normalize_separator_characters(str)));
    }

    template <class V>
    bool try_extract_vector(std::string& str, V& out)
    {
        std::string vector_str;
        if (try_extract_vector_sub_string(str, vector_str)&& V::try_parse(vector_str, out))
            return true;

        return false;
    }

    template <class V>
    std::vector<V> extract_vectors(const std::string& str)
    {
        std::vector<V> result;
        auto str_copy = str;
        V out;
        while (try_extract_vector(str_copy, out))
            result.push_back(out);

        return result;
    }

    template <class V>
    V extract_vector(const std::string& str)
    {
        auto str_copy = str;
        V result;
        if (!try_extract_vector(str_copy, result))
            throw std::exception("Failed to parse a vector");

        return result;
    }

    template <class V>
    std::string to_string(const std::vector<V>& collection, const char delim)
    {
        std::string result = "";
        for (const auto& item : collection)
            result += item.to_string() + delim;

        return result;
    }

    std::vector<std::string> split_by_char(const std::string& str, const char delim)
    {
        std::stringstream ss(str);
        std::string part;
        std::vector<std::string> result;

        while (std::getline(ss, part, delim))
            result.push_back(part);

        return result;
    }

    std::string read_all_text(const std::filesystem::path& file_name)
    {
        std::ifstream input_file(file_name);
        if (!input_file.is_open()) {
            throw std::exception("Can't open file");
        }

        const auto text = std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());

        return text;
    }

    std::string create_guid_string()
    {
        GUID guid;

        if (S_OK == CoCreateGuid(&guid))
        {
            constexpr int buffer_size = 37;
            std::vector<char> buffer(buffer_size);

            if (snprintf(buffer.data(), buffer.size(),
                "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX",
                guid.Data1, guid.Data2, guid.Data3,
                guid.Data4[0], guid.Data4[1], guid.Data4[2],
                guid.Data4[3], guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]) != buffer_size - 1)
                throw std::exception("snprintf has failed");
            return {buffer.data()};
        }

        throw std::exception("Failed to create GUID");
    }

    /// <summary>
    /// Returns dd_hh:mm:ss string representation of the given time duration in milliseconds
    ///	`SecRatio` template parameter specifies number of units of the given time argument in 1 second;
    ///	i.e., 1 if the argument represents number of seconds,
    ///	1000 if the argument represents number of milliseconds,
    ///	1000000 if the argument represents number of microseconds and so on...
    /// </summary>
    template <int SecRatio = 1>
    std::string time_to_dd_hh_mm_ss_string(const long long& time)
    {
        //extract number of full days as the `chrono` method below can't handle values that exceed 24 h
        constexpr auto units_in_day = 3600 * 24 * SecRatio;
        const auto days = time / units_in_day;
        const auto time_normalized = time % units_in_day;
        std::stringstream ss;
        if (days > 0)
            ss << days << "d:";

        ss << std::chrono::hh_mm_ss(std::chrono::duration<int64_t, std::ratio<1, SecRatio>>(time_normalized));
        return ss.str();
    }

    std::string seconds_to_dd_hh_mm_ss_string(const long long& time_sec)
    {
        return time_to_dd_hh_mm_ss_string(time_sec);
    }

    std::string milliseconds_to_dd_hh_mm_ss_string(const long long& time_msec)
    {
        return time_to_dd_hh_mm_ss_string<1000>(time_msec);
    }

    std::string format_date_time(const std::chrono::system_clock::time_point& time_pt)
    {
        std::time_t t = std::chrono::system_clock::to_time_t(time_pt);
        char str[100];

        if (ctime_s(str, sizeof str, &t) != 0)
            throw std::exception("Can't convert time to string");

        return str;
    }

    std::string get_elapsed_time_formatted(const std::chrono::system_clock::time_point& start_pt, const std::chrono::system_clock::time_point& stop_pt)
    {
        const auto epoch_time_sec = std::chrono::duration_cast<std::chrono::seconds>(stop_pt - start_pt);
        return std::format("{:%T}", epoch_time_sec);
    }

    std::string get_elapsed_time_formatted(std::chrono::system_clock::time_point& start_pt)
    {
        const auto stop_pt = std::chrono::system_clock::now();
        const auto result = get_elapsed_time_formatted(start_pt, stop_pt);
        start_pt = stop_pt;
        return result;
    }

    template bool try_extract_vector(std::string& str, Vector2d<Real>& out);
    template bool try_extract_vector(std::string& str, Vector3d<Real>& out);
    template bool try_extract_vector(std::string& str, Index2d& out);
    template bool try_extract_vector(std::string& str, Index3d& out);

    template std::vector<Vector3d<Real>> extract_vectors(const std::string& str);
    template std::vector<Vector2d<Real>> extract_vectors(const std::string& str);
    template std::vector<Index3d> extract_vectors(const std::string& str);
    template std::vector<Index2d> extract_vectors(const std::string& str);

    template Vector3d<Real> extract_vector(const std::string& str);
    template Vector2d<Real> extract_vector(const std::string& str);
    template Index3d extract_vector(const std::string& str);
    template Index2d extract_vector(const std::string& str);

    template std::string to_string(const std::vector<Vector3d<Real>>& collection, const char delim);
    template std::string to_string(const std::vector<Vector2d<Real>>& collection, const char delim);
    template std::string to_string(const std::vector<Index3d>& collection, const char delim);
    template std::string to_string(const std::vector<Index2d>& collection, const char delim);
}
