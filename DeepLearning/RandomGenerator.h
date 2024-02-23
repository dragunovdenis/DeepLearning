//Copyright (c) 2024 Denys Dragunov, dragunovdenis@gmail.com
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
#include "defs.h"

namespace DeepLearning
{
    /// <summary>
    /// Functionality to generate pseudorandom numbers.
    /// </summary>
    class RandomGenerator {
        std::mt19937 _gen;
        std::uniform_real_distribution<Real> _dist;
    public:
        /// <summary>
        /// Constructor.
        /// </summary>
        RandomGenerator(const unsigned int seed = std::random_device{}());

        /// <summary>
        /// Returns pseudo-random floating point number from [0, 1).
        /// </summary>
        Real next();

        /// <summary>
        /// Returns pseudo-random integer number from the suggested range
        /// (left boundary is included, right boundary is excluded).
        /// </summary>
        int get_int(const int min_included, const int max_excluded);
    };
}
