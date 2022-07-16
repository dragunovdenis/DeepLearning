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
#include "../CudaBridge.h"
#include<cmath>

namespace DeepLearning::Func
{
    /// <summary>
     /// A "cuda-callable" max function
     /// </summary>
    template<class T>
    CUDA_CALLABLE T cuda_max(const T& a, const T& b)
    {
        return a > b ? a : b;
    }

    /// <summary>
    /// A "cuda-callable" min function
    /// </summary>
    template<class T>
    CUDA_CALLABLE T cuda_min(const T& a, const T& b)
    {
        return a < b ? a : b;
    }

    /// <summary>
    /// Sigmoid function
    /// </summary>
    template <class T>
    CUDA_CALLABLE T sigmoid(const T& arg)
    {
        return T(1) / (T(1) + exp(-arg));
    }

    /// <summary>
    /// An analogous of the Python's nan_to_num() function
    /// </summary>
    template <class R>
    CUDA_CALLABLE R nan_to_num(const R& val) {
        if (isinf(val)) {
            if (val < R(0))
                return -std::numeric_limits<R>::max();
            else
                return std::numeric_limits<R>::max();
        }
        else if (isnan(val)) {
            return R(0);
        }

        return val;
    }
}