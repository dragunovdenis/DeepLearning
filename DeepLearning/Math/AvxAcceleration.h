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
#include <cstddef>
#include <immintrin.h>
#include <type_traits>

namespace DeepLearning::Avx
{
	/// <summary>
	/// Calculate dot product of the given pair of vectors using 4-doubles operations
	/// </summary>
	/// <param name="vec1">Pointer to the beginning of the first vector</param>
	/// <param name="vec2">Pointer to the beginning of the second vector</param>
	/// <param name="size">Size of the vectors</param>
	/// <returns>Dot product of the vectors</returns>
	double mm256_dot_product(const double* vec1, const double* vec2, const std::size_t size);

	/// <summary>
	/// Calculate dot product of the given pair of vectors using 8-floats operations
	/// </summary>
	/// <param name="vec1">Pointer to the beginning of the first vector</param>
	/// <param name="vec2">Pointer to the beginning of the second vector</param>
	/// <param name="size">Size of the vectors</param>
	/// <returns>Dot product of the vectors</returns>
	float mm256_dot_product(const float* vec1, const float* vec2, const std::size_t size);

	/// <summary>
	/// SIMD-accelerated scaled-add: dst[i] += scale * src[i] for i in [0, n).
	/// </summary>
	template <typename T>
	inline void scaled_add(T* dst, const T* src, const T scale, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			const __m256d s = _mm256_set1_pd(scale);
			for (; i + 4 <= n; i += 4)
			{
				const __m256d d = _mm256_loadu_pd(dst + i);
				const __m256d v = _mm256_loadu_pd(src + i);
				_mm256_storeu_pd(dst + i, _mm256_add_pd(d, _mm256_mul_pd(v, s)));
			}
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			const __m256 s = _mm256_set1_ps(scale);
			for (; i + 8 <= n; i += 8)
			{
				const __m256 d = _mm256_loadu_ps(dst + i);
				const __m256 v = _mm256_loadu_ps(src + i);
				_mm256_storeu_ps(dst + i, _mm256_add_ps(d, _mm256_mul_ps(v, s)));
			}
		}
		for (; i < n; ++i) dst[i] += scale * src[i];
	}
}