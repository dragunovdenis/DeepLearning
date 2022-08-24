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

namespace DeepLearning::Utils
{
    Real get_random(const Real min, const Real max)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

        const auto normalization_factor = (max - min) / std::numeric_limits<int>::max();
        return normalization_factor * dist(gen) + min;
    }

    int get_random_int(const int min, const int max)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

        const auto interval_length = (static_cast<long long>(max) - min) + 1;
        return dist(gen) % interval_length + min;
    }

    std::string extract_word(std::string& str)
    {
        const auto word_start_pos = str.find_first_not_of(' ');

        if (word_start_pos == std::string::npos) // there are no words
        {
            str = std::string();
            return std::string();
        }

        auto word_end_pos = str.find_first_of(' ', word_start_pos);

        if (word_end_pos == std::string::npos)
            word_end_pos = str.length();

        const auto result = str.substr(word_start_pos, word_end_pos - word_start_pos);

        str.erase(0, word_end_pos);

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

    std::string remove_leading_trailing_extra_spaces(const std::string& str)
    {
        return std::regex_replace(str, std::regex("^ +| +$|( ) +"), "$1");
    }

    std::string to_upper_case(const std::string& str)
    {
        auto result = str;
        std::transform(str.begin(), str.end(), result.begin(), ::toupper);

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

    template bool try_extract_vector(std::string& str, Vector2d<Real>& out);
    template bool try_extract_vector(std::string& str, Vector3d<Real>& out);
    template bool try_extract_vector(std::string& str, Index2d& out);
    template bool try_extract_vector(std::string& str, Index3d& out);

    template std::vector<Vector3d<Real>> extract_vectors(const std::string& str);
    template std::vector<Vector2d<Real>> extract_vectors(const std::string& str);
    template std::vector<Index3d> extract_vectors(const std::string& str);
    template std::vector<Index2d> extract_vectors(const std::string& str);

    template std::string to_string(const std::vector<Vector3d<Real>>& collection, const char delim);
    template std::string to_string(const std::vector<Vector2d<Real>>& collection, const char delim);
    template std::string to_string(const std::vector<Index3d>& collection, const char delim);
    template std::string to_string(const std::vector<Index2d>& collection, const char delim);
}