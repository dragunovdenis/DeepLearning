#include "DenseVector.h"
#include "../Utilities.h"
#include <exception>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

namespace DeepLearning
{
	DenseVector::DenseVector(const std::size_t dim)
	{
		_data.resize(dim, Real(0));
	}

	DenseVector::DenseVector(const std::size_t dim, const Real range_begin, const Real range_end)
		: DenseVector(dim)
	{
		Utils::fill_with_random_values(begin(), end(), range_begin, range_end);
	}

	bool DenseVector::check_bounds(const ::std::size_t id) const
	{
		return id < _data.size();
	}

	std::size_t DenseVector::dim() const
	{
		return _data.size();
	}

	Real& DenseVector::operator ()(const std::size_t id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(row_id, col_id))
			throw std::exception("Index out of bounds")
#endif // CHECK_BOUNDS

			return _data[id];
	}

	const Real& DenseVector::operator ()(const std::size_t id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(row_id, col_id))
			throw std::exception("Index out of bounds")
#endif // CHECK_BOUNDS

			return _data[id];
	}

	bool DenseVector::operator ==(const DenseVector& vect) const
	{
		return _data == vect._data;
	}

	bool DenseVector::operator !=(const DenseVector& vect) const
	{
		return !(*this == vect);
	}

	std::vector<Real>::iterator DenseVector::begin()
	{
		return _data.begin();
	}

	std::vector<Real>::const_iterator DenseVector::begin() const
	{
		return _data.begin();
	}

	std::vector<Real>::iterator DenseVector::end()
	{
		return _data.end();
	}

	std::vector<Real>::const_iterator DenseVector::end() const
	{
		return _data.end();
	}

	static DenseVector random(const std::size_t dim, const Real range_begin, const Real range_end)
	{
		auto result = DenseVector(dim);
		Utils::fill_with_random_values(result.begin(), result.end(), range_begin, range_end);

		return result;
	}

	DenseVector& DenseVector::operator += (const DenseVector& vec)
	{
		if (vec.dim() != dim())
			throw std::exception("Operands must be of the same dimension");

		std::transform(begin(), end(), vec.begin(), begin(), [](const auto& x, const auto& y) { return x + y; });
		return *this;
	}

	DenseVector& DenseVector::operator -= (const DenseVector& vec)
	{
		if (vec.dim() != dim())
			throw std::exception("Operands must be of the same dimension");

		std::transform(begin(), end(), vec.begin(), begin(), [](const auto& x, const auto& y) { return x - y; });
		return *this;
	}

	DenseVector& DenseVector::operator *= (const Real& scalar)
	{
		std::transform(begin(), end(), begin(), [scalar](const auto& x) { return x * scalar; });
		return *this;
	}

	DenseVector operator +(const DenseVector& vec1, const DenseVector& vec2)
	{
		auto result = vec1;
		return result += vec2;
	}

	DenseVector operator -(const DenseVector& vec1, const DenseVector& vec2)
	{
		auto result = vec1;
		return result -= vec2;
	}

	DenseVector operator *(const DenseVector& vec, const Real& scalar)
	{
		auto result = vec;
		return result *= scalar;
	}

	DenseVector operator *(const Real& scalar, const DenseVector& vec)
	{
		return vec * scalar;
	}

	Real DenseVector::max_abs() const
	{
		return std::abs(*std::max_element(begin(), end(), [](const auto& x, const auto& y) { return std::abs(x) < std::abs(y); }));
	}

	void DenseVector::fill(const Real& val)
	{
		std::fill(begin(), end(), val);
	}

	DenseVector DenseVector::hadamard_prod(const DenseVector& vec) const
	{
		DenseVector result(dim());
		std::transform(begin(), end(), vec.begin(), result.begin(), [](const auto& x, const auto& y) { return x * y; });
		return result;
	}

}