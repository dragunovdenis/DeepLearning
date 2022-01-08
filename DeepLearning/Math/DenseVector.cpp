#include "DenseVector.h"
#include "../Utilities.h"
#include <exception>
#include <stdlib.h>
#include <time.h>

namespace DeepLearning
{
	DenseVector::DenseVector(const std::size_t dim)
	{
		_data.resize(dim, Real(0));
	}

	DenseVector::DenseVector(const std::size_t dim, const Real range_begin, const Real range_end)
		: DenseVector(dim)
	{
		fill_with_random_values(begin(), end(), range_begin, range_end);
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
		fill_with_random_values(result.begin(), result.end(), range_begin, range_end);

		return result;
	}
}