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

#include "AvxAcceleration.h"
#include <immintrin.h>


namespace DeepLearning::Avx
{
	/// <summary>
	/// Sums up given 4 doubles and returns the result
	/// </summary>
	double mm256_reduce(const __m256d& input) {
		const auto temp = _mm256_hadd_pd(input, input);
		const auto sum_high = _mm256_extractf128_pd(temp, 1);
		const auto result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(temp));
		return ((double*)&result)[0];
	}

	/// <summary>
	/// Sums up given 8 floats and returns the result
	/// </summary>
	float mm256_reduce(const __m256& input) {
		const auto t1 = _mm256_hadd_ps(input, input);
		const auto t2 = _mm256_hadd_ps(t1, t1);
		const auto t3 = _mm256_extractf128_ps(t2, 1);
		const auto t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
		return _mm_cvtss_f32(t4);
	}

	double mm256_dot_product(const double* vec1, const double* vec2, const std::size_t size) {
		auto sum_vec = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

		/* Add up partial dot-products in blocks of 256 bits */
		const auto chunk_size = 4;
		const auto chunks_count = size / chunk_size;
		std::size_t offset = 0;
		for (std::size_t chunk_id = 0; chunk_id < chunks_count; chunk_id++) {
			const auto x = _mm256_loadu_pd(vec1 + offset);
			const auto y = _mm256_loadu_pd(vec2 + offset);
			offset += chunk_size;
			sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(x, y));
		}

		/* Find the partial dot-product for the remaining elements after
		 * dealing with all 256-bit blocks. */
		double rest = 0.0;
		for (std::size_t element_id = size - (size % chunk_size); element_id < size; element_id++)
			rest += vec1[element_id] * vec2[element_id];

		return mm256_reduce(sum_vec) + rest;
	}

	float mm256_dot_product(const float* vec1, const float* vec2, const std::size_t size) {
		auto sum_vec = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

		/* Add up partial dot-products in blocks of 256 bits */
		const auto chunk_size = 8;
		const auto chunks_count = size / chunk_size;
		std::size_t offset = 0;
		for (std::size_t chunk_id = 0; chunk_id < chunks_count; chunk_id++) {
			const auto x = _mm256_loadu_ps(vec1 + offset);
			const auto y = _mm256_loadu_ps(vec2 + offset);
			offset += chunk_size;
			sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x, y));
		}

		/* Find the partial dot-product for the remaining elements after
		 * dealing with all 256-bit blocks. */
		float rest = 0.0;
		for (std::size_t element_id = size - (size % chunk_size); element_id < size; element_id++)
			rest += vec1[element_id] * vec2[element_id];

		return mm256_reduce(sum_vec) + rest;
	}

}