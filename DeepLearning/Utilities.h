#pragma once

#include <random>
#include <algorithm>
#include "defs.h"

namespace DeepLearning::Utils
{
    /// <summary>
    /// Functionality to fill the given range with uniformly distributed pseudo-random numbers
    /// </summary>
    template< class Iter>
    void fill_with_random_values(Iter start, Iter end, const Real min, const Real max)
    {
        static std::random_device rd;
        static std::mt19937 mte(rd());

        std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

        const auto normalization_factor = (max - min) / std::numeric_limits<int>::max();

        std::generate(start, end, [&]() { return normalization_factor * dist(mte) + min; });
    }

    /// <summary>
    /// Converts given value to string with the given number of digits
    /// </summary>
    template <typename T>
    std::string to_string(const T a_value, const int n = std::numeric_limits<T>::digits10)
    {
        std::ostringstream out;
        out.precision(n);
        out << std::fixed << a_value;
        return out.str();
    }
}
