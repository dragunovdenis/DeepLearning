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

#include "DenseMatrix.h"
#include "DenseVector.h"
#include "../Utilities.h"
#include <exception>
#include <numeric>
#include <iterator>
#include <cstddef>
#include <algorithm>
#include <immintrin.h>

#define USE_AVX2 //to use AVX2 instructions below

namespace DeepLearning
{
	/// <summary>
	/// An iterator allowing to traverse columns of the matrix
	/// </summary>
	template <typename T>
	struct ColumnIterator
	{
		using iterator_category = std::input_iterator_tag;
		using difference_type   = std::ptrdiff_t;
		using value_type		= T;
		using pointer			= T*;
		using reference			= T&;

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="ptr">Pointer</param>
		/// <param name="col_dim">Offset that will be used to calculate position of the "next" pointer</param>
		ColumnIterator(pointer ptr, const std::size_t col_dim) :_ptr(ptr), _col_dim(col_dim){}

		reference operator *() const { return *_ptr; }
		pointer operator ->() { return _ptr; }

		/// <summary>
		/// Prefix increment
		/// </summary>
		ColumnIterator& operator++() {
			_ptr += _col_dim; 
			return *this;
		}

		/// <summary>
		/// Postfix increment
		/// </summary>
		ColumnIterator operator ++(int) { 
			auto temp = *this; 
			++(*this); 
			return temp; 
		}

		friend bool operator== (const ColumnIterator& a, const ColumnIterator& b) { return a._ptr == b._ptr && a._col_dim == b._col_dim; };
		friend bool operator!= (const ColumnIterator& a, const ColumnIterator& b) { return a._ptr != b._ptr || a._col_dim != b._col_dim; };

	private:

		std::size_t _col_dim;
		pointer _ptr;

	};

	std::size_t DenseMatrix::size() const
	{
		return _row_dim * _col_dim;
	}

	bool DenseMatrix::check_bounds(const std::size_t row_id, const std::size_t col_id) const
	{
		return row_id < _row_dim && col_id < _col_dim;
	}

	DenseMatrix::DenseMatrix(const std::size_t row_dim, const std::size_t col_dim):_row_dim(row_dim), _col_dim(col_dim)
	{
		_data.resize(size(), Real(0));
	}

	DenseMatrix::DenseMatrix(const std::size_t row_dim, const std::size_t col_dim,
		const std::function<Real()>& generator) : DenseMatrix(row_dim, col_dim)
	{
		std::generate(begin(), end(), generator);
	}

	DenseMatrix::DenseMatrix(const std::size_t row_dim, const std::size_t col_dim,
		const Real range_begin, const Real range_end) : DenseMatrix(row_dim, col_dim)
	{
		Utils::fill_with_random_values(begin(), end(), range_begin, range_end);
	}

	std::size_t DenseMatrix::row_col_to_data_id(const std::size_t row_id, const std::size_t col_id) const
	{
		return row_id * _col_dim + col_id;
	}

	Real& DenseMatrix::operator ()(const std::size_t row_id, const std::size_t col_id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(row_id, col_id))
			throw std::exception("Index out of bounds")
#endif // CHECK_BOUNDS

		return _data[row_col_to_data_id(row_id, col_id)];
	}

	const Real& DenseMatrix::operator ()(const std::size_t row_id, const std::size_t col_id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(row_id, col_id))
			throw std::exception("Index out of bounds")
#endif // CHECK_BOUNDS

		return _data[row_col_to_data_id(row_id, col_id)];
	}

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
	/// Calculate dot product of the given pair of vectors using 4-doubles operations
	/// </summary>
	/// <param name="vec1">Pointer to the beginning of the first vector</param>
	/// <param name="vec2">Pointer to the beginning of the second vector</param>
	/// <param name="size">Size of the vectors</param>
	/// <returns>Dot product of the vectors</returns>
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

	DenseVector operator *(const DenseMatrix& matr, const DenseVector& vec)
	{
		if (vec.dim() != matr._col_dim)
			throw std::exception("Incompatible matrix-vector dimensionality");

		auto result = DenseVector(matr._row_dim);

		for (std::size_t row_id = 0; row_id < matr._row_dim; row_id++)
		{
#ifdef USE_AVX2
			const auto begin_row_ptr = matr._data.data() + row_id * matr._col_dim;
			result(row_id) = mm256_dot_product(begin_row_ptr, &*vec.begin(), vec.dim());
#else
			const auto row_begin = matr._data.begin() + row_id * matr._col_dim;
			const auto row_end = matr._data.begin() + row_id * matr._col_dim + matr._col_dim;
			result(row_id) = std::inner_product(row_begin, row_end, vec.begin(), Real(0));
#endif // USE_AVX2
		}

		return result;
	}

	DenseVector DenseMatrix::mul_add(const DenseVector& mul_vec, const DenseVector& add_vec) const
	{
		if (mul_vec.dim() != _col_dim || add_vec.dim() != _row_dim)
			throw std::exception("Incompatible matrix-vector dimensionality");

		auto result = DenseVector(_row_dim);

		for (std::size_t row_id = 0; row_id < row_dim(); row_id++)
		{
#ifdef USE_AVX2
			const auto begin_row_ptr = _data.data() + row_id * _col_dim;
			result(row_id) = mm256_dot_product(begin_row_ptr, &*mul_vec.begin(), _col_dim) + add_vec(row_id);
#else
			const auto row_begin = _data.begin() + row_id * col_dim();
			const auto row_end = _data.begin() + (row_id + 1) * _col_dim;
			result(row_id) = std::inner_product(row_begin, row_end, mul_vec.begin(), Real(0)) + +add_vec(row_id);
#endif // USE_AVX2
		}

		return result;
	}

	DenseVector operator *(const DenseVector& vec, const DenseMatrix& matr)
	{
		if (vec.dim() != matr._row_dim)
			throw std::exception("Incompatible matrix-vector dimension");

		auto result = DenseVector(matr._col_dim);

		const auto last_row_offset = matr._col_dim * (matr._row_dim - 1);

		for (std::size_t col_id = 0; col_id < matr._col_dim; col_id++)
		{
			const auto col_begin = ColumnIterator(&*(matr.begin() + col_id), matr._col_dim);
			const auto col_end = ++ColumnIterator(&*(matr.begin() + last_row_offset + col_id), matr._col_dim);
			result(col_id) = std::inner_product(col_begin, col_end, vec.begin(), Real(0));
		}

		return result;
	}

	bool DenseMatrix::operator ==(const DenseMatrix& matr) const
	{
		return _row_dim == matr._row_dim &&
			   _col_dim == matr._col_dim &&
		       _data == matr._data;
	}

	bool DenseMatrix::operator !=(const DenseMatrix& matr) const
	{
		return !(*this == matr);
	}

	std::vector<Real>::iterator DenseMatrix::begin()
	{
		return _data.begin();
	}

	std::vector<Real>::const_iterator DenseMatrix::begin() const
	{
		return _data.begin();
	}

	std::vector<Real>::iterator DenseMatrix::end()
	{
		return _data.end();
	}

	std::vector<Real>::const_iterator DenseMatrix::end() const
	{
		return _data.end();
	}

	static inline DenseMatrix random(const std::size_t row_dim, const std::size_t col_dim, const Real range_begin, const Real range_end)
	{
		auto result = DenseMatrix(row_dim, col_dim);
		Utils::fill_with_random_values(result.begin(), result.end(), range_begin, range_end);

		return result;
	}

	DenseMatrix& DenseMatrix::operator +=(const DenseMatrix& mat)
	{
		if (_row_dim != mat._row_dim || _col_dim != mat._col_dim)
			throw std::exception("Operands must be of the same dimension");

		std::transform(begin(), end(), mat.begin(), begin(), [](const auto& x, const auto& y) { return x + y; });
		return *this;
	}

	DenseMatrix& DenseMatrix::operator -=(const DenseMatrix& mat)
	{
		if (_row_dim != mat._row_dim || _col_dim != mat._col_dim)
			throw std::exception("Operands must be of the same dimension");

		std::transform(begin(), end(), mat.begin(), begin(), [](const auto& x, const auto& y) { return x - y; });
		return *this;
	}

	DenseMatrix& DenseMatrix::operator *=(const Real& scalar)
	{
		std::transform(begin(), end(), begin(), [scalar](const auto& x) { return x * scalar; });
		return *this;
	}

	DenseMatrix operator +(const DenseMatrix& mat1, const DenseMatrix& mat2)
	{
		auto result = mat1;
		return result += mat2;
	}

	/// <summary>
	/// Matrix subtraction operator
	/// </summary>
	DenseMatrix operator -(const DenseMatrix& mat1, const DenseMatrix& mat2)
	{
		auto result = mat1;
		return result -= mat2;
	}

	/// <summary>
	/// Matrix by scalar multiplication operator
	/// </summary>
	DenseMatrix operator *(const DenseMatrix& mat, const Real& scalar)
	{
		auto result = mat;
		return result *= scalar;

	}

	/// <summary>
	/// Scalar by matrix multiplication operator
	/// </summary>
	DenseMatrix operator *(const Real& scalar, const DenseMatrix& mat)
	{
		return mat * scalar;
	}

	Real DenseMatrix::max_abs() const
	{
		return std::abs(*std::max_element(begin(), end(), [](const auto& x, const auto& y) { return std::abs(x) < std::abs(y); }));
	}

	void DenseMatrix::fill(const Real& val)
	{
		std::fill(begin(), end(), val);
	}

	std::size_t DenseMatrix::col_dim() const
	{
		return _col_dim;
	}

	std::size_t DenseMatrix::row_dim() const
	{
		return _row_dim;
	}

	DenseMatrix vector_col_times_vector_row(const DenseVector& vec_col, const DenseVector& vec_row)
	{
		DenseMatrix result(vec_col.dim(), vec_row.dim());

		const auto col_dim = result.col_dim();
		const auto last_row_offset = col_dim * (result.row_dim() - 1);

		for (std::size_t col_id = 0; col_id < vec_row.dim(); col_id++)
		{
			const auto col_begin = ColumnIterator(&*(result.begin() + col_id), col_dim);
			const auto factor = vec_row(col_id);
			std::transform(vec_col.begin(), vec_col.end(), col_begin, [factor](const auto& x) {return factor * x; });
		}

		return result;
	}

}