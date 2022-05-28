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
#include <regex>

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

        const auto interval_length = (max - min) + 1;
        return dist(gen) % interval_length + min;
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
}