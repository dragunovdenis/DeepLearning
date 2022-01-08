#include "DenseMatrix.h"
#include "DenseVector.h"
#include "../Utilities.h"
#include <exception>
#include <numeric>
#include <iterator>
#include <cstddef>

namespace DeepLearning
{
	/// <summary>
	/// An iterator allowing to traverse columns of the matrix
	/// </summary>
	struct ColumnIterator
	{
		using iterator_category = std::input_iterator_tag;
		using difference_type   = std::ptrdiff_t;
		using value_type        = Real;
		using pointer           = const Real*;
		using reference         = const Real&;

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
		const Real range_begin, const Real range_end) : DenseMatrix(row_dim, col_dim)
	{
		fill_with_random_values(begin(), end(), range_begin, range_end);
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
			throw std::exception("Incompatible matrix-vector dimension");

		auto result = DenseVector(matr._row_dim);

		for (std::size_t row_id = 0; row_id < matr._row_dim; row_id++)
		{
			const auto row_begin = matr._data.begin() + row_id * matr._col_dim;
			const auto row_end = matr._data.begin() + row_id * matr._col_dim + matr._col_dim;
			result(row_id) = std::inner_product(row_begin, row_end, vec.begin(), Real(0));
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
			const auto col_begin = ColumnIterator(&(matr._data[col_id]), matr._col_dim);
			const auto col_end = ++ColumnIterator(&(matr._data[last_row_offset + col_id]), matr._col_dim);
			result(col_id) = std::inner_product(col_begin, col_end, vec.begin(), Real(0));
		}

		return result;
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
		fill_with_random_values(result.begin(), result.end(), range_begin, range_end);

		return result;
	}
}