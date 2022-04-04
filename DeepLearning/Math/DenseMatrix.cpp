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
#include "AvxAcceleration.h"
#include "../IndexIterator.h"

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

	DenseMatrix::DenseMatrix(const std::size_t row_dim, const std::size_t col_dim, const bool assign_zero)
		:_row_dim(row_dim), _col_dim(col_dim)
	{
		_data = reinterpret_cast<Real*>(std::malloc(size() * sizeof(Real)));

		if (assign_zero)
			std::fill(begin(), end(), Real(0));
	}

	DenseMatrix::DenseMatrix(const std::size_t row_dim, const std::size_t col_dim,
		const std::function<Real()>& generator) : DenseMatrix(row_dim, col_dim, false)
	{
		std::generate(begin(), end(), generator);
	}

	DenseMatrix::DenseMatrix(const std::size_t row_dim, const std::size_t col_dim,
		const Real range_begin, const Real range_end) : DenseMatrix(row_dim, col_dim, false)
	{
		Utils::fill_with_random_values(begin(), end(), range_begin, range_end);
	}

	DenseMatrix::DenseMatrix(const DenseMatrix& matr) : DenseMatrix(matr.row_dim(), matr.col_dim(), false)
	{
		std::copy(matr.begin(), matr.end(), begin());
	}

	DenseMatrix& DenseMatrix::operator =(const DenseMatrix& matr)
	{
		if (size() != matr.size())
		{
			free();
			_data = reinterpret_cast<Real*>(std::malloc(matr.size() * sizeof(Real)));
		}

		_col_dim = matr.col_dim();
		_row_dim = matr.row_dim();

		std::copy(matr.begin(), matr.end(), begin());

		return *this;
	}

	DenseMatrix::DenseMatrix(DenseMatrix&& matr) noexcept 
		: _col_dim(matr._col_dim), _row_dim(matr._row_dim), _data(matr._data)
	{
		matr._data = nullptr;
		matr._col_dim = 0;
		matr._row_dim = 0;
	}

	void DenseMatrix::free()
	{
		if (_data != nullptr)
		{
			delete[] _data;
			_data = nullptr;
		}

		_col_dim = 0;
		_row_dim = 0;
	}

	DenseMatrix::~DenseMatrix()
	{
		free();
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

	DenseVector operator *(const DenseMatrix& matr, const DenseVector& vec)
	{
		if (vec.dim() != matr._col_dim)
			throw std::exception("Incompatible matrix-vector dimensionality");

		auto result = DenseVector(matr._row_dim);

		for (std::size_t row_id = 0; row_id < matr._row_dim; row_id++)
		{
#ifdef USE_AVX2
			const auto begin_row_ptr = matr._data + row_id * matr._col_dim;
			result(row_id) = Avx::mm256_dot_product(begin_row_ptr, &*vec.begin(), vec.dim());
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
			const auto begin_row_ptr = _data + row_id * _col_dim;
			result(row_id) = Avx::mm256_dot_product(begin_row_ptr, &*mul_vec.begin(), _col_dim) + add_vec(row_id);
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
			std::all_of(IndexIterator(0), IndexIterator(static_cast<int>(size())),
				[&](const auto id) { return _data[id] == matr._data[id]; });
	}

	bool DenseMatrix::operator !=(const DenseMatrix& matr) const
	{
		return !(*this == matr);
	}

	Real* DenseMatrix::begin()
	{
		return _data;
	}

	const Real* DenseMatrix::begin() const
	{
		return _data;
	}

	Real* DenseMatrix::end()
	{
		return _data + size();
	}

	const Real* DenseMatrix::end() const
	{
		return _data + size();
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

		add(mat);
		return *this;
	}

	DenseMatrix& DenseMatrix::operator -=(const DenseMatrix& mat)
	{
		if (_row_dim != mat._row_dim || _col_dim != mat._col_dim)
			throw std::exception("Operands must be of the same dimension");

		sub(mat);
		return *this;
	}

	DenseMatrix& DenseMatrix::operator *=(const Real& scalar)
	{
		mul(scalar);
		return *this;
	}

	DenseMatrix operator +(const DenseMatrix& mat1, const DenseMatrix& mat2)
	{
		auto result = mat1;
		return result += mat2;
	}

	DenseMatrix operator -(const DenseMatrix& mat1, const DenseMatrix& mat2)
	{
		auto result = mat1;
		return result -= mat2;
	}

	DenseMatrix operator *(const DenseMatrix& mat, const Real& scalar)
	{
		auto result = mat;
		return result *= scalar;

	}

	DenseMatrix operator *(const Real& scalar, const DenseMatrix& mat)
	{
		return mat * scalar;
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