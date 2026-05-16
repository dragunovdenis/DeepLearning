//Copyright (c) 2026 Denys Dragunov, dragunovdenis@gmail.com
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
	/// Horizontal sum of a 256-bit register of doubles.
	/// </summary>
	inline double mm256_reduce(const __m256d& input) {
		const __m128d lo = _mm256_castpd256_pd128(input);
		const __m128d hi = _mm256_extractf128_pd(input, 1);
		const __m128d s = _mm_add_pd(lo, hi);
		const __m128d shuf = _mm_unpackhi_pd(s, s);
		return _mm_cvtsd_f64(_mm_add_sd(s, shuf));
	}

	/// <summary>
	/// Horizontal sum of a 256-bit register of floats.
	/// </summary>
	inline float mm256_reduce(const __m256& input) {
		const __m128 lo = _mm256_castps256_ps128(input);
		const __m128 hi = _mm256_extractf128_ps(input, 1);
		__m128 s = _mm_add_ps(lo, hi);
		s = _mm_add_ps(s, _mm_movehl_ps(s, s));
		s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
		return _mm_cvtss_f32(s);
	}

	/// <summary>
	/// Dot product (double) using FMA and 4 independent accumulators to break
	/// the latency-bound dependency chain. Inline to ensure cross-TU inlining
	/// (the non-inlined call was visible as a separate symbol in CPU profiles).
	/// </summary>
	/// <param name="vec1">Pointer to the beginning of the first vector</param>
	/// <param name="vec2">Pointer to the beginning of the second vector</param>
	/// <param name="size">Size of the vectors</param>
	/// <returns>Dot product of the vectors</returns>
	inline double mm256_dot_product(const double* vec1, const double* vec2, const std::size_t size) {
		__m256d acc0 = _mm256_setzero_pd();
		__m256d acc1 = _mm256_setzero_pd();
		__m256d acc2 = _mm256_setzero_pd();
		__m256d acc3 = _mm256_setzero_pd();

		constexpr std::size_t lane = 4;          // doubles per 256-bit register
		constexpr std::size_t step = 4 * lane;   // 16 doubles per unrolled iteration

		std::size_t i = 0;
		for (; i + step <= size; i += step) {
			const __m256d x0 = _mm256_loadu_pd(vec1 + i + 0 * lane);
			const __m256d y0 = _mm256_loadu_pd(vec2 + i + 0 * lane);
			const __m256d x1 = _mm256_loadu_pd(vec1 + i + 1 * lane);
			const __m256d y1 = _mm256_loadu_pd(vec2 + i + 1 * lane);
			const __m256d x2 = _mm256_loadu_pd(vec1 + i + 2 * lane);
			const __m256d y2 = _mm256_loadu_pd(vec2 + i + 2 * lane);
			const __m256d x3 = _mm256_loadu_pd(vec1 + i + 3 * lane);
			const __m256d y3 = _mm256_loadu_pd(vec2 + i + 3 * lane);
			acc0 = _mm256_fmadd_pd(x0, y0, acc0);
			acc1 = _mm256_fmadd_pd(x1, y1, acc1);
			acc2 = _mm256_fmadd_pd(x2, y2, acc2);
			acc3 = _mm256_fmadd_pd(x3, y3, acc3);
		}

		// Tail of 4-lane chunks
		for (; i + lane <= size; i += lane) {
			const __m256d x = _mm256_loadu_pd(vec1 + i);
			const __m256d y = _mm256_loadu_pd(vec2 + i);
			acc0 = _mm256_fmadd_pd(x, y, acc0);
		}

		// Pair-tree reduction of the four accumulators (lower rounding error
		// than left-fold; also matches the multi-accumulator dataflow).
		const __m256d s01 = _mm256_add_pd(acc0, acc1);
		const __m256d s23 = _mm256_add_pd(acc2, acc3);
		const __m256d s = _mm256_add_pd(s01, s23);
		double result = mm256_reduce(s);

		// Scalar tail (< 4 elements)
		for (; i < size; ++i) result += vec1[i] * vec2[i];

		return result;
	}

	/// <summary>
	/// Dot product (float) using FMA and 4 independent accumulators.
	/// </summary>
	/// <param name="vec1">Pointer to the beginning of the first vector</param>
	/// <param name="vec2">Pointer to the beginning of the second vector</param>
	/// <param name="size">Size of the vectors</param>
	/// <returns>Dot product of the vectors</returns>
	inline float mm256_dot_product(const float* vec1, const float* vec2, const std::size_t size) {
		__m256 acc0 = _mm256_setzero_ps();
		__m256 acc1 = _mm256_setzero_ps();
		__m256 acc2 = _mm256_setzero_ps();
		__m256 acc3 = _mm256_setzero_ps();

		constexpr std::size_t lane = 8;          // floats per 256-bit register
		constexpr std::size_t step = 4 * lane;   // 32 floats per unrolled iteration

		std::size_t i = 0;
		for (; i + step <= size; i += step) {
			const __m256 x0 = _mm256_loadu_ps(vec1 + i + 0 * lane);
			const __m256 y0 = _mm256_loadu_ps(vec2 + i + 0 * lane);
			const __m256 x1 = _mm256_loadu_ps(vec1 + i + 1 * lane);
			const __m256 y1 = _mm256_loadu_ps(vec2 + i + 1 * lane);
			const __m256 x2 = _mm256_loadu_ps(vec1 + i + 2 * lane);
			const __m256 y2 = _mm256_loadu_ps(vec2 + i + 2 * lane);
			const __m256 x3 = _mm256_loadu_ps(vec1 + i + 3 * lane);
			const __m256 y3 = _mm256_loadu_ps(vec2 + i + 3 * lane);
			acc0 = _mm256_fmadd_ps(x0, y0, acc0);
			acc1 = _mm256_fmadd_ps(x1, y1, acc1);
			acc2 = _mm256_fmadd_ps(x2, y2, acc2);
			acc3 = _mm256_fmadd_ps(x3, y3, acc3);
		}

		// Tail of 8-lane chunks
		for (; i + lane <= size; i += lane) {
			const __m256 x = _mm256_loadu_ps(vec1 + i);
			const __m256 y = _mm256_loadu_ps(vec2 + i);
			acc0 = _mm256_fmadd_ps(x, y, acc0);
		}

		const __m256 s01 = _mm256_add_ps(acc0, acc1);
		const __m256 s23 = _mm256_add_ps(acc2, acc3);
		const __m256 s = _mm256_add_ps(s01, s23);
		float result = mm256_reduce(s);

		for (; i < size; ++i) result += vec1[i] * vec2[i];

		return result;
	}

	/// <summary>
	/// SIMD-accelerated scaled-add: dst[i] += scale * src[i] for i in [0, n).
	/// Uses FMA and a 4-way unrolled main loop to break the latency-bound
	/// dependency chain, matching the shape of the simd_* helpers below.
	/// </summary>
	template <typename T>
	inline void scaled_add(T* dst, const T* src, const T scale, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			const __m256d s = _mm256_set1_pd(scale);
			constexpr std::size_t lane = 4;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_pd(dst + i + 0 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(src + i + 0 * lane), s, _mm256_loadu_pd(dst + i + 0 * lane)));
				_mm256_storeu_pd(dst + i + 1 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(src + i + 1 * lane), s, _mm256_loadu_pd(dst + i + 1 * lane)));
				_mm256_storeu_pd(dst + i + 2 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(src + i + 2 * lane), s, _mm256_loadu_pd(dst + i + 2 * lane)));
				_mm256_storeu_pd(dst + i + 3 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(src + i + 3 * lane), s, _mm256_loadu_pd(dst + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_pd(dst + i, _mm256_fmadd_pd(_mm256_loadu_pd(src + i), s, _mm256_loadu_pd(dst + i)));
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			const __m256 s = _mm256_set1_ps(scale);
			constexpr std::size_t lane = 8;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_ps(dst + i + 0 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 0 * lane), s, _mm256_loadu_ps(dst + i + 0 * lane)));
				_mm256_storeu_ps(dst + i + 1 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 1 * lane), s, _mm256_loadu_ps(dst + i + 1 * lane)));
				_mm256_storeu_ps(dst + i + 2 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 2 * lane), s, _mm256_loadu_ps(dst + i + 2 * lane)));
				_mm256_storeu_ps(dst + i + 3 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 3 * lane), s, _mm256_loadu_ps(dst + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_loadu_ps(src + i), s, _mm256_loadu_ps(dst + i)));
		}
		for (; i < n; ++i) dst[i] += scale * src[i];
	}

	// Common shape for the simd_* helpers below: a 4-way unrolled AVX2 main loop
	// (for double, lane=4 and step=16; for float, lane=8 and step=32) that issues
	// 4 independent AVX2 instructions per unrolled iteration to hide
	// load/store/ALU latency, followed by a single-vector tail, then a scalar
	// tail. Kept as a non-Doxygen comment so it doesn't bind to simd_add.

	/// <summary>SIMD elementwise add: dst[i] += src[i].</summary>
	template <typename T>
	inline void simd_add(T* dst, const T* src, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			constexpr std::size_t lane = 4;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_pd(dst + i + 0 * lane, _mm256_add_pd(_mm256_loadu_pd(dst + i + 0 * lane), _mm256_loadu_pd(src + i + 0 * lane)));
				_mm256_storeu_pd(dst + i + 1 * lane, _mm256_add_pd(_mm256_loadu_pd(dst + i + 1 * lane), _mm256_loadu_pd(src + i + 1 * lane)));
				_mm256_storeu_pd(dst + i + 2 * lane, _mm256_add_pd(_mm256_loadu_pd(dst + i + 2 * lane), _mm256_loadu_pd(src + i + 2 * lane)));
				_mm256_storeu_pd(dst + i + 3 * lane, _mm256_add_pd(_mm256_loadu_pd(dst + i + 3 * lane), _mm256_loadu_pd(src + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_pd(dst + i, _mm256_add_pd(_mm256_loadu_pd(dst + i), _mm256_loadu_pd(src + i)));
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			constexpr std::size_t lane = 8;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_ps(dst + i + 0 * lane, _mm256_add_ps(_mm256_loadu_ps(dst + i + 0 * lane), _mm256_loadu_ps(src + i + 0 * lane)));
				_mm256_storeu_ps(dst + i + 1 * lane, _mm256_add_ps(_mm256_loadu_ps(dst + i + 1 * lane), _mm256_loadu_ps(src + i + 1 * lane)));
				_mm256_storeu_ps(dst + i + 2 * lane, _mm256_add_ps(_mm256_loadu_ps(dst + i + 2 * lane), _mm256_loadu_ps(src + i + 2 * lane)));
				_mm256_storeu_ps(dst + i + 3 * lane, _mm256_add_ps(_mm256_loadu_ps(dst + i + 3 * lane), _mm256_loadu_ps(src + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
		}
		for (; i < n; ++i) dst[i] += src[i];
	}

	/// <summary>SIMD elementwise sub: dst[i] -= src[i].</summary>
	template <typename T>
	inline void simd_sub(T* dst, const T* src, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			constexpr std::size_t lane = 4;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_pd(dst + i + 0 * lane, _mm256_sub_pd(_mm256_loadu_pd(dst + i + 0 * lane), _mm256_loadu_pd(src + i + 0 * lane)));
				_mm256_storeu_pd(dst + i + 1 * lane, _mm256_sub_pd(_mm256_loadu_pd(dst + i + 1 * lane), _mm256_loadu_pd(src + i + 1 * lane)));
				_mm256_storeu_pd(dst + i + 2 * lane, _mm256_sub_pd(_mm256_loadu_pd(dst + i + 2 * lane), _mm256_loadu_pd(src + i + 2 * lane)));
				_mm256_storeu_pd(dst + i + 3 * lane, _mm256_sub_pd(_mm256_loadu_pd(dst + i + 3 * lane), _mm256_loadu_pd(src + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_pd(dst + i, _mm256_sub_pd(_mm256_loadu_pd(dst + i), _mm256_loadu_pd(src + i)));
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			constexpr std::size_t lane = 8;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_ps(dst + i + 0 * lane, _mm256_sub_ps(_mm256_loadu_ps(dst + i + 0 * lane), _mm256_loadu_ps(src + i + 0 * lane)));
				_mm256_storeu_ps(dst + i + 1 * lane, _mm256_sub_ps(_mm256_loadu_ps(dst + i + 1 * lane), _mm256_loadu_ps(src + i + 1 * lane)));
				_mm256_storeu_ps(dst + i + 2 * lane, _mm256_sub_ps(_mm256_loadu_ps(dst + i + 2 * lane), _mm256_loadu_ps(src + i + 2 * lane)));
				_mm256_storeu_ps(dst + i + 3 * lane, _mm256_sub_ps(_mm256_loadu_ps(dst + i + 3 * lane), _mm256_loadu_ps(src + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
		}
		for (; i < n; ++i) dst[i] -= src[i];
	}

	/// <summary>SIMD in-place scale: dst[i] *= scale.</summary>
	template <typename T>
	inline void simd_mul_scalar(T* dst, const T scale, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			const __m256d s = _mm256_set1_pd(scale);
			constexpr std::size_t lane = 4;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_pd(dst + i + 0 * lane, _mm256_mul_pd(_mm256_loadu_pd(dst + i + 0 * lane), s));
				_mm256_storeu_pd(dst + i + 1 * lane, _mm256_mul_pd(_mm256_loadu_pd(dst + i + 1 * lane), s));
				_mm256_storeu_pd(dst + i + 2 * lane, _mm256_mul_pd(_mm256_loadu_pd(dst + i + 2 * lane), s));
				_mm256_storeu_pd(dst + i + 3 * lane, _mm256_mul_pd(_mm256_loadu_pd(dst + i + 3 * lane), s));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(dst + i), s));
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			const __m256 s = _mm256_set1_ps(scale);
			constexpr std::size_t lane = 8;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_ps(dst + i + 0 * lane, _mm256_mul_ps(_mm256_loadu_ps(dst + i + 0 * lane), s));
				_mm256_storeu_ps(dst + i + 1 * lane, _mm256_mul_ps(_mm256_loadu_ps(dst + i + 1 * lane), s));
				_mm256_storeu_ps(dst + i + 2 * lane, _mm256_mul_ps(_mm256_loadu_ps(dst + i + 2 * lane), s));
				_mm256_storeu_ps(dst + i + 3 * lane, _mm256_mul_ps(_mm256_loadu_ps(dst + i + 3 * lane), s));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(dst + i), s));
		}
		for (; i < n; ++i) dst[i] *= scale;
	}

	/// <summary>SIMD scale-and-add: dst[i] = dst[i] * scale + src[i].</summary>
	template <typename T>
	inline void simd_scale_and_add(T* dst, const T* src, const T scale, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			const __m256d s = _mm256_set1_pd(scale);
			constexpr std::size_t lane = 4;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_pd(dst + i + 0 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i + 0 * lane), s, _mm256_loadu_pd(src + i + 0 * lane)));
				_mm256_storeu_pd(dst + i + 1 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i + 1 * lane), s, _mm256_loadu_pd(src + i + 1 * lane)));
				_mm256_storeu_pd(dst + i + 2 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i + 2 * lane), s, _mm256_loadu_pd(src + i + 2 * lane)));
				_mm256_storeu_pd(dst + i + 3 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i + 3 * lane), s, _mm256_loadu_pd(src + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_pd(dst + i, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i), s, _mm256_loadu_pd(src + i)));
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			const __m256 s = _mm256_set1_ps(scale);
			constexpr std::size_t lane = 8;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_ps(dst + i + 0 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 0 * lane), s, _mm256_loadu_ps(src + i + 0 * lane)));
				_mm256_storeu_ps(dst + i + 1 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 1 * lane), s, _mm256_loadu_ps(src + i + 1 * lane)));
				_mm256_storeu_ps(dst + i + 2 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 2 * lane), s, _mm256_loadu_ps(src + i + 2 * lane)));
				_mm256_storeu_ps(dst + i + 3 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 3 * lane), s, _mm256_loadu_ps(src + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i), s, _mm256_loadu_ps(src + i)));
		}
		for (; i < n; ++i) dst[i] = dst[i] * scale + src[i];
	}

	/// <summary>SIMD axpby: dst[i] = dst[i] * scale_0 + src[i] * scale_1.</summary>
	template <typename T>
	inline void simd_scale_and_add_scaled(T* dst, const T* src, const T scale_0, const T scale_1, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			const __m256d s0 = _mm256_set1_pd(scale_0);
			const __m256d s1 = _mm256_set1_pd(scale_1);
			constexpr std::size_t lane = 4;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_pd(dst + i + 0 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i + 0 * lane), s0, _mm256_mul_pd(_mm256_loadu_pd(src + i + 0 * lane), s1)));
				_mm256_storeu_pd(dst + i + 1 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i + 1 * lane), s0, _mm256_mul_pd(_mm256_loadu_pd(src + i + 1 * lane), s1)));
				_mm256_storeu_pd(dst + i + 2 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i + 2 * lane), s0, _mm256_mul_pd(_mm256_loadu_pd(src + i + 2 * lane), s1)));
				_mm256_storeu_pd(dst + i + 3 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i + 3 * lane), s0, _mm256_mul_pd(_mm256_loadu_pd(src + i + 3 * lane), s1)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_pd(dst + i, _mm256_fmadd_pd(_mm256_loadu_pd(dst + i), s0, _mm256_mul_pd(_mm256_loadu_pd(src + i), s1)));
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			const __m256 s0 = _mm256_set1_ps(scale_0);
			const __m256 s1 = _mm256_set1_ps(scale_1);
			constexpr std::size_t lane = 8;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_ps(dst + i + 0 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 0 * lane), s0, _mm256_mul_ps(_mm256_loadu_ps(src + i + 0 * lane), s1)));
				_mm256_storeu_ps(dst + i + 1 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 1 * lane), s0, _mm256_mul_ps(_mm256_loadu_ps(src + i + 1 * lane), s1)));
				_mm256_storeu_ps(dst + i + 2 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 2 * lane), s0, _mm256_mul_ps(_mm256_loadu_ps(src + i + 2 * lane), s1)));
				_mm256_storeu_ps(dst + i + 3 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 3 * lane), s0, _mm256_mul_ps(_mm256_loadu_ps(src + i + 3 * lane), s1)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i), s0, _mm256_mul_ps(_mm256_loadu_ps(src + i), s1)));
		}
		for (; i < n; ++i) dst[i] = dst[i] * scale_0 + src[i] * scale_1;
	}

	/// <summary>SIMD Hadamard product: dst[i] = a[i] * b[i].</summary>
	template <typename T>
	inline void simd_hadamard(T* dst, const T* a, const T* b, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			constexpr std::size_t lane = 4;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_pd(dst + i + 0 * lane, _mm256_mul_pd(_mm256_loadu_pd(a + i + 0 * lane), _mm256_loadu_pd(b + i + 0 * lane)));
				_mm256_storeu_pd(dst + i + 1 * lane, _mm256_mul_pd(_mm256_loadu_pd(a + i + 1 * lane), _mm256_loadu_pd(b + i + 1 * lane)));
				_mm256_storeu_pd(dst + i + 2 * lane, _mm256_mul_pd(_mm256_loadu_pd(a + i + 2 * lane), _mm256_loadu_pd(b + i + 2 * lane)));
				_mm256_storeu_pd(dst + i + 3 * lane, _mm256_mul_pd(_mm256_loadu_pd(a + i + 3 * lane), _mm256_loadu_pd(b + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			constexpr std::size_t lane = 8;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_ps(dst + i + 0 * lane, _mm256_mul_ps(_mm256_loadu_ps(a + i + 0 * lane), _mm256_loadu_ps(b + i + 0 * lane)));
				_mm256_storeu_ps(dst + i + 1 * lane, _mm256_mul_ps(_mm256_loadu_ps(a + i + 1 * lane), _mm256_loadu_ps(b + i + 1 * lane)));
				_mm256_storeu_ps(dst + i + 2 * lane, _mm256_mul_ps(_mm256_loadu_ps(a + i + 2 * lane), _mm256_loadu_ps(b + i + 2 * lane)));
				_mm256_storeu_ps(dst + i + 3 * lane, _mm256_mul_ps(_mm256_loadu_ps(a + i + 3 * lane), _mm256_loadu_ps(b + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
		}
		for (; i < n; ++i) dst[i] = a[i] * b[i];
	}

	/// <summary>SIMD fused Hadamard-add: dst[i] += a[i] * b[i].</summary>
	template <typename T>
	inline void simd_hadamard_add(T* dst, const T* a, const T* b, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			constexpr std::size_t lane = 4;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_pd(dst + i + 0 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 0 * lane), _mm256_loadu_pd(b + i + 0 * lane), _mm256_loadu_pd(dst + i + 0 * lane)));
				_mm256_storeu_pd(dst + i + 1 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 1 * lane), _mm256_loadu_pd(b + i + 1 * lane), _mm256_loadu_pd(dst + i + 1 * lane)));
				_mm256_storeu_pd(dst + i + 2 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 2 * lane), _mm256_loadu_pd(b + i + 2 * lane), _mm256_loadu_pd(dst + i + 2 * lane)));
				_mm256_storeu_pd(dst + i + 3 * lane, _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 3 * lane), _mm256_loadu_pd(b + i + 3 * lane), _mm256_loadu_pd(dst + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_pd(dst + i, _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), _mm256_loadu_pd(dst + i)));
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			constexpr std::size_t lane = 8;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				_mm256_storeu_ps(dst + i + 0 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 0 * lane), _mm256_loadu_ps(b + i + 0 * lane), _mm256_loadu_ps(dst + i + 0 * lane)));
				_mm256_storeu_ps(dst + i + 1 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 1 * lane), _mm256_loadu_ps(b + i + 1 * lane), _mm256_loadu_ps(dst + i + 1 * lane)));
				_mm256_storeu_ps(dst + i + 2 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 2 * lane), _mm256_loadu_ps(b + i + 2 * lane), _mm256_loadu_ps(dst + i + 2 * lane)));
				_mm256_storeu_ps(dst + i + 3 * lane, _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 3 * lane), _mm256_loadu_ps(b + i + 3 * lane), _mm256_loadu_ps(dst + i + 3 * lane)));
			}
			for (; i + lane <= n; i += lane)
				_mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_loadu_ps(dst + i)));
		}
		for (; i < n; ++i) dst[i] += a[i] * b[i];
	}

	/// <summary>SIMD sum of squares: sum(src[i]^2). Uses 4 independent
	/// accumulators with FMA to break the latency-bound dependency chain.</summary>
	template <typename T>
	inline T simd_sum_of_squares(const T* src, const std::size_t n)
	{
		std::size_t i = 0;
		if constexpr (std::is_same_v<T, double>)
		{
			__m256d acc0 = _mm256_setzero_pd();
			__m256d acc1 = _mm256_setzero_pd();
			__m256d acc2 = _mm256_setzero_pd();
			__m256d acc3 = _mm256_setzero_pd();
			constexpr std::size_t lane = 4;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				const __m256d x0 = _mm256_loadu_pd(src + i + 0 * lane);
				const __m256d x1 = _mm256_loadu_pd(src + i + 1 * lane);
				const __m256d x2 = _mm256_loadu_pd(src + i + 2 * lane);
				const __m256d x3 = _mm256_loadu_pd(src + i + 3 * lane);
				acc0 = _mm256_fmadd_pd(x0, x0, acc0);
				acc1 = _mm256_fmadd_pd(x1, x1, acc1);
				acc2 = _mm256_fmadd_pd(x2, x2, acc2);
				acc3 = _mm256_fmadd_pd(x3, x3, acc3);
			}
			for (; i + lane <= n; i += lane)
			{
				const __m256d x = _mm256_loadu_pd(src + i);
				acc0 = _mm256_fmadd_pd(x, x, acc0);
			}
			const __m256d s01 = _mm256_add_pd(acc0, acc1);
			const __m256d s23 = _mm256_add_pd(acc2, acc3);
			double result = mm256_reduce(_mm256_add_pd(s01, s23));
			for (; i < n; ++i) result += src[i] * src[i];
			return result;
		}
		else if constexpr (std::is_same_v<T, float>)
		{
			__m256 acc0 = _mm256_setzero_ps();
			__m256 acc1 = _mm256_setzero_ps();
			__m256 acc2 = _mm256_setzero_ps();
			__m256 acc3 = _mm256_setzero_ps();
			constexpr std::size_t lane = 8;
			constexpr std::size_t step = 4 * lane;
			for (; i + step <= n; i += step)
			{
				const __m256 x0 = _mm256_loadu_ps(src + i + 0 * lane);
				const __m256 x1 = _mm256_loadu_ps(src + i + 1 * lane);
				const __m256 x2 = _mm256_loadu_ps(src + i + 2 * lane);
				const __m256 x3 = _mm256_loadu_ps(src + i + 3 * lane);
				acc0 = _mm256_fmadd_ps(x0, x0, acc0);
				acc1 = _mm256_fmadd_ps(x1, x1, acc1);
				acc2 = _mm256_fmadd_ps(x2, x2, acc2);
				acc3 = _mm256_fmadd_ps(x3, x3, acc3);
			}
			for (; i + lane <= n; i += lane)
			{
				const __m256 x = _mm256_loadu_ps(src + i);
				acc0 = _mm256_fmadd_ps(x, x, acc0);
			}
			const __m256 s01 = _mm256_add_ps(acc0, acc1);
			const __m256 s23 = _mm256_add_ps(acc2, acc3);
			float result = mm256_reduce(_mm256_add_ps(s01, s23));
			for (; i < n; ++i) result += src[i] * src[i];
			return result;
		}
		else
		{
			T result{};
			for (; i < n; ++i) result += src[i] * src[i];
			return result;
		}
	}
}