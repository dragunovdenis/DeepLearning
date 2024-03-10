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

#include "Matrix.h"
#include "Tensor.h"
#include "Vector.h"
#include "../Utilities.h"
#include <exception>
#include <numeric>
#include <iterator>
#include <cstddef>
#include <algorithm>
#include "AvxAcceleration.h"
#include "../IndexIterator.h"
#include "../Diagnostics/Logging.h"

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
		ColumnIterator(pointer ptr, const std::size_t col_dim) : _col_dim(col_dim), _ptr(ptr){}

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

	void Matrix::resize(const std::size_t& new_row_dim, const std::size_t& new_col_dim)
	{
		const auto new_size = new_row_dim * new_col_dim;
		if (_capacity < new_size)
		{
			free();
			_data = reinterpret_cast<Real*>(std::malloc(new_size * sizeof(Real)));
			_capacity = new_size;
		}

		_row_dim = new_row_dim;
		_col_dim = new_col_dim;
	}

	void Matrix::resize(const Index3d& size_3d)
	{
		if (size_3d.x != 1ll)
			throw std::exception("Invalid input data");

		resize(size_3d.y, size_3d.z);
	}

	std::size_t Matrix::size() const
	{
		return _row_dim * _col_dim;
	}

	std::size_t Matrix::capacity() const
	{
		return _capacity;
	}

	Index3d Matrix::size_3d() const
	{
		return { 1ull, _row_dim, _col_dim };
	}

	bool Matrix::check_bounds(const std::size_t row_id, const std::size_t col_id) const
	{
		return row_id < _row_dim && col_id < _col_dim;
	}

	Matrix::Matrix(const std::size_t row_dim, const std::size_t col_dim, const bool assign_zero)
	{
		resize(row_dim, col_dim);

		if (assign_zero)
			std::fill(begin(), end(), Real(0));
	}

	Matrix::Matrix(const Index3d& size, const bool assign_zero) :
		Matrix(size.y, size.z, assign_zero)
	{
		if (size.x != 1ll)
			throw std::exception("Invalid input size");
	}

	Matrix::Matrix(const std::size_t row_dim, const std::size_t col_dim,
		const Real range_begin, const Real range_end, std::mt19937* seeder) :
		Matrix(row_dim, col_dim, false)
	{
		uniform_random_fill(range_begin, range_end, seeder);
	}

	Matrix::Matrix(const Matrix& matr) : Matrix(matr.row_dim(), matr.col_dim(), false)
	{
		std::copy(matr.begin(), matr.end(), begin());
	}

	Matrix& Matrix::operator =(const Matrix& matr)
	{
		if (this != &matr)
		{
			resize(matr.row_dim(), matr.col_dim());
			std::copy(matr.begin(), matr.end(), begin());
		}

		return *this;
	}

	void Matrix::abandon_resources()
	{
		_data = nullptr;
		free();
	}

	Matrix::Matrix(Matrix&& matr) noexcept 
		: _row_dim(matr._row_dim), _col_dim(matr._col_dim), _capacity(matr._capacity)
	{
		_data = matr._data;
		matr.abandon_resources();
	}

	Matrix& Matrix::operator =(Matrix&& matr) noexcept
	{
		if (this != &matr)
		{
			free();
			_col_dim = matr._col_dim;
			_row_dim = matr._row_dim;
			_capacity = matr._capacity;
			_data = matr._data;
			matr.abandon_resources();
		}

		return *this;
	}

	void Matrix::free()
	{
		if (_data != nullptr)
		{
			delete[] _data;
			_data = nullptr;
		}

		_col_dim = 0;
		_row_dim = 0;
		_capacity = 0;
	}

	Matrix::~Matrix()
	{
		free();
	}

	std::size_t Matrix::row_col_to_data_id(const std::size_t row_id, const std::size_t col_id) const
	{
		return row_id * _col_dim + col_id;
	}

	Real& Matrix::operator ()(const std::size_t row_id, const std::size_t col_id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(row_id, col_id))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return _data[row_col_to_data_id(row_id, col_id)];
	}

	const Real& Matrix::operator ()(const std::size_t row_id, const std::size_t col_id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(row_id, col_id))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return _data[row_col_to_data_id(row_id, col_id)];
	}

	Vector operator *(const Matrix& matr, const BasicCollection& vec)
	{
		if (vec.size() != matr._col_dim)
			throw std::exception("Incompatible matrix-vector dimensionality");

		auto result = Vector(matr._row_dim);

		for (std::size_t row_id = 0; row_id < matr._row_dim; row_id++)
		{
#ifdef USE_AVX2
			const auto begin_row_ptr = matr._data + row_id * matr._col_dim;
			result(row_id) = Avx::mm256_dot_product(begin_row_ptr, vec.begin(), vec.size());
#else
			const auto row_begin = matr._data.begin() + row_id * matr._col_dim;
			const auto row_end = matr._data.begin() + row_id * matr._col_dim + matr._col_dim;
			result(row_id) = std::inner_product(row_begin, row_end, vec.begin(), Real(0));
#endif // USE_AVX2
		}

		return result;
	}

	Vector Matrix::mul_add(const BasicCollection& mul_vec, const BasicCollection& add_vec) const
	{
		auto result = Vector(_row_dim, false /*assign zero*/);
		mul_add(mul_vec, add_vec, result);
		return result;
	}

	void Matrix::mul_add(const BasicCollection& mul_vec, const BasicCollection& add_vec, BasicCollection& result) const
	{
		if (mul_vec.size() != _col_dim || add_vec.size() != _row_dim || result.size() != _row_dim)
			throw std::exception("Incompatible matrix-vector dimensionality");

		for (std::size_t row_id = 0; row_id < row_dim(); row_id++)
		{
#ifdef USE_AVX2
			const auto begin_row_ptr = _data + row_id * _col_dim;
			result.begin()[row_id] = Avx::mm256_dot_product(begin_row_ptr, mul_vec.begin(), _col_dim) + add_vec[row_id];
#else
			const auto row_begin = _data.begin() + row_id * col_dim();
			const auto row_end = _data.begin() + (row_id + 1) * _col_dim;
			result.begin()[row_id] = std::inner_product(row_begin, row_end, mul_vec.begin(), Real(0)) + add_vec[row_id];
#endif // USE_AVX2
		}
	}

	void Matrix::transpose_mul(const BasicCollection& vec, BasicCollection& result) const
	{
		if (vec.size() != _row_dim)
			throw std::exception("Incompatible matrix-vector dimension");

		result.resize({ 1ull, 1ull, _col_dim });
		result.fill(0);

		for (std::size_t row_id = 0; row_id < _row_dim; ++row_id)
		{
			const auto scalar = vec[row_id];

			std::transform(result.begin(), result.end(), begin() + row_id * _col_dim, result.begin(),
				[scalar](const auto& x, const auto& y) { return y * scalar + x; });
		}
	}

	Vector operator *(const BasicCollection& vec, const Matrix& matr)
	{
		Vector result;
		matr.transpose_mul(vec, result);
		return result;
	}

	Matrix Matrix::transpose() const
	{
		Matrix result;
		transpose(result);
		return result;
	}

	void Matrix::transpose(Matrix& out) const
	{
		out.resize({ 1ull, _col_dim, _row_dim });

		for (auto row_id = 0ull; row_id < _row_dim; ++row_id)
			for (auto col_id = 0ull; col_id < _col_dim; ++col_id)
				out._data[col_id * _row_dim + row_id] = _data[row_id * _col_dim + col_id];
	}

	bool Matrix::operator ==(const Matrix& matr) const
	{
		return _row_dim == matr._row_dim &&
			   _col_dim == matr._col_dim &&
			std::all_of(IndexIterator(0), IndexIterator(static_cast<int>(size())),
				[&](const auto id) { return _data[id] == matr._data[id]; });
	}

	bool Matrix::operator !=(const Matrix& matr) const
	{
		return !(*this == matr);
	}

	[[maybe_unused]] static inline Matrix random(const std::size_t row_dim, const std::size_t col_dim, const Real range_begin, const Real range_end)
	{
		auto result = Matrix(row_dim, col_dim);
		Utils::fill_with_random_values(result.begin(), result.end(), range_begin, range_end);

		return result;
	}

	Matrix& Matrix::operator +=(const Matrix& mat)
	{
		if (_row_dim != mat._row_dim || _col_dim != mat._col_dim)
			throw std::exception("Operands must be of the same dimension");

		add(mat);
		return *this;
	}

	Matrix& Matrix::operator -=(const Matrix& mat)
	{
		if (_row_dim != mat._row_dim || _col_dim != mat._col_dim)
			throw std::exception("Operands must be of the same dimension");

		sub(mat);
		return *this;
	}

	Matrix& Matrix::operator *=(const Real& scalar)
	{
		mul(scalar);
		return *this;
	}

	Matrix operator +(const Matrix& mat1, const Matrix& mat2)
	{
		auto result = mat1;
		return result += mat2;
	}

	Matrix operator -(const Matrix& mat1, const Matrix& mat2)
	{
		auto result = mat1;
		return result -= mat2;
	}

	Matrix operator *(const Matrix& mat, const Real& scalar)
	{
		auto result = mat;
		return result *= scalar;

	}

	Matrix operator *(const Real& scalar, const Matrix& mat)
	{
		return mat * scalar;
	}

	std::size_t Matrix::col_dim() const
	{
		return _col_dim;
	}

	std::size_t Matrix::row_dim() const
	{
		return _row_dim;
	}

	template <class T>
	void vector_col_times_vector_row(const BasicCollection& vec_col, const BasicCollection& vec_row, T& result)
	{
		result.resize({ 1ull, vec_col.size(), vec_row.size() });
		const auto col_dim = result.col_dim();

		for (std::size_t row_id = 0; row_id < vec_col.size(); row_id++)
		{
			const auto factor = vec_col[row_id];
			std::transform(vec_row.begin(), vec_row.end(),
				result.begin() + col_dim * row_id, [factor](const auto& x) {return factor * x; });
		}
	}

	template void vector_col_times_vector_row<Matrix>(const BasicCollection& vec_col, const BasicCollection& vec_row, Matrix& result);
	template void vector_col_times_vector_row<Tensor>(const BasicCollection& vec_col, const BasicCollection& vec_row, Tensor& result);

	Matrix vector_col_times_vector_row(const BasicCollection& vec_col, const BasicCollection& vec_row)
	{
		Matrix result;
		vector_col_times_vector_row(vec_col, vec_row, result);
		return result;
	}

	void Matrix::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		std::vector<Real> proxy;
		std::size_t row_dim, col_dim;
		msgpack::type::make_define_array(row_dim, col_dim, proxy).msgpack_unpack(msgpack_o);
		resize(row_dim, col_dim);
		std::copy(proxy.begin(), proxy.end(), begin());
	}

	void Matrix::log(const std::filesystem::path& filename) const
	{
		Logging::log_as_table(get_handle(), row_dim(), col_dim(), filename);
	}
}