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

#include "DenseVector.h"
#include "../Utilities.h"
#include <exception>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include "../IndexIterator.h"

namespace DeepLearning
{
	DenseVector::DenseVector(const DenseVector& vec)
		: DenseVector(vec.dim(), false)
	{
		std::copy(vec.begin(), vec.end(), begin());
	}

	DenseVector& DenseVector::operator=(const DenseVector& vec)
	{
		assign(vec);
		return *this;
	}

	DenseVector::DenseVector(DenseVector&& vec) noexcept : _dim(vec._dim), _data(vec._data)
	{
		vec._data = nullptr;
		vec._dim = 0;
	}

	DenseVector::DenseVector(const std::size_t dim, const bool assign_zero) : _dim(dim)
	{
		_data = reinterpret_cast<Real*>(std::malloc(dim * sizeof(Real)));

		if (assign_zero)
			std::fill(begin(), end(), Real(0));
	}

	template <class T>
	DenseVector::DenseVector(const std::vector<T>& source)
		: DenseVector(source.size(), false)
	{
		std::copy(source.begin(), source.end(), begin());
	}

	DenseVector::DenseVector(const std::size_t dim, const Real range_begin, const Real range_end)
		: DenseVector(dim, false)
	{
		Utils::fill_with_random_values(begin(), end(), range_begin, range_end);
	}

	DenseVector::DenseVector(const std::size_t dim, const std::function<Real()>& generator)
		: DenseVector(dim, false)
	{
		std::generate(begin(), end(), generator);
	}

	DenseVector::~DenseVector()
	{
		free();
	}

	/// <summary>
	/// Frees the allocated memory
	/// </summary>
	void DenseVector::free()
	{
		if (_data != nullptr)
		{
			delete[] _data;
			_data = nullptr;
		}
		_dim = 0;
	}

	std::vector<Real> DenseVector::ToStdVector() const
	{
		return std::vector<Real>(begin(), end());
	}

	template <class S>
	void DenseVector::assign(const S& source)
	{
		if (size() != source.size())
		{
			free();
			_dim = source.size();
			_data = reinterpret_cast<Real*>(std::malloc(_dim * sizeof(Real)));
		}
		std::copy(source.begin(), source.end(), begin());
	}

	template void DenseVector::assign(const std::vector<Real>& source);
	template void DenseVector::assign(const DenseVector& source);

	bool DenseVector::check_bounds(const ::std::size_t id) const
	{
		return id < _dim;
	}

	std::size_t DenseVector::dim() const
	{
		return _dim;
	}

	std::size_t DenseVector::size() const
	{
		return _dim;
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
		return _dim == vect._dim && std::all_of(IndexIterator(0), IndexIterator(static_cast<int>(_dim)),
			[&](const auto& index) { return _data[index] == vect._data[index]; });
	}

	bool DenseVector::operator !=(const DenseVector& vect) const
	{
		return !(*this == vect);
	}

	Real* DenseVector::begin()
	{
		return _data;
	}

	const Real* DenseVector::begin() const
	{
		return _data;
	}

	Real* DenseVector::end()
	{
		return _data + _dim;
	}

	const Real* DenseVector::end() const
	{
		return _data + _dim;
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

	std::size_t DenseVector::max_element_id(const std::function<bool(Real, Real)>& comparer) const
	{
		const auto id = std::max_element(begin(), end(), comparer) - begin();
		return static_cast<std::size_t>(id);
	}

	Real DenseVector::sum(const std::function<Real(Real)>& transform_operator) const
	{
		return std::accumulate(begin(), end(), Real(0), [&transform_operator](const auto& sum, const auto& x) { return sum + transform_operator(x); });
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

	template DenseVector::DenseVector(const std::vector<unsigned char>& souurce);
	template DenseVector::DenseVector(const std::vector<Real>& souurce);
}