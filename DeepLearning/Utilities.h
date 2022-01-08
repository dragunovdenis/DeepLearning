#pragma once

#include <random>
#include <algorithm>
#include "defs.h"

namespace DeepLearning
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
}
