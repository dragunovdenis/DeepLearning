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

#include "Vector.h"
#include "../Utilities.h"
#include <exception>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include "../IndexIterator.h"
#include "..//Diagnostics/Logging.h"

namespace DeepLearning
{
	Vector::Vector(const Vector& vec)
		: Vector(vec.dim(), false)
	{
		std::copy(vec.begin(), vec.end(), begin());
	}

	Vector& Vector::operator=(const Vector& vec)
	{
		assign(vec);
		return *this;
	}

	void Vector::abandon_resources()
	{
		_data = nullptr;
		free();
	}

	Vector::Vector(Vector&& vec) noexcept : _dim(vec._dim)
	{
		_data = vec._data;
		vec.abandon_resources();
	}

	Vector::Vector(const std::size_t dim, const bool assign_zero) : _dim(dim)
	{
		_data = reinterpret_cast<Real*>(std::malloc(dim * sizeof(Real)));

		if (assign_zero)
			std::fill(begin(), end(), Real(0));
	}

	Vector::Vector(const Index3d& size, const bool assign_zero) :
		Vector(size.z, assign_zero)
	{
		if (size.x != 1ll || size.y != 1ll)
			throw std::exception("Invalid input size");
	}

	template <class T>
	Vector::Vector(const std::vector<T>& source)
		: Vector(source.size(), false)
	{
		std::copy(source.begin(), source.end(), begin());
	}

	Vector::Vector(const std::size_t dim, const Real range_begin, const Real range_end)
		: Vector(dim, false)
	{
		Utils::fill_with_random_values(begin(), end(), range_begin, range_end);
	}

	Vector::~Vector()
	{
		free();
	}

	/// <summary>
	/// Frees the allocated memory
	/// </summary>
	void Vector::free()
	{
		if (_data != nullptr)
		{
			delete[] _data;
			_data = nullptr;
		}
		_dim = 0;
	}

	template <class S>
	void Vector::assign(const S& source)
	{
		if (size() != source.size())
		{
			free();
			_dim = source.size();
			_data = reinterpret_cast<Real*>(std::malloc(_dim * sizeof(Real)));
		}
		std::copy(source.begin(), source.end(), begin());
	}

	template void Vector::assign(const std::vector<Real>& source);
	template void Vector::assign(const Vector& source);

	bool Vector::check_bounds(const ::std::size_t id) const
	{
		return id < _dim;
	}

	std::size_t Vector::dim() const
	{
		return _dim;
	}

	std::size_t Vector::size() const
	{
		return _dim;
	}

	Index3d Vector::size_3d() const
	{
		return { 1ull, 1ull, _dim };
	}

	Real& Vector::operator ()(const std::size_t id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(id))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

			return _data[id];
	}

	const Real& Vector::operator ()(const std::size_t id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(id))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

			return _data[id];
	}

	bool Vector::operator ==(const Vector& vect) const
	{
		return _dim == vect._dim && std::all_of(IndexIterator(0), IndexIterator(static_cast<int>(_dim)),
			[&](const auto& index) { return _data[index] == vect._data[index]; });
	}

	bool Vector::operator !=(const Vector& vect) const
	{
		return !(*this == vect);
	}

	static Vector random(const std::size_t dim, const Real range_begin, const Real range_end)
	{
		auto result = Vector(dim);
		Utils::fill_with_random_values(result.begin(), result.end(), range_begin, range_end);

		return result;
	}

	Vector& Vector::operator += (const Vector& vec)
	{
		if (vec.dim() != dim())
			throw std::exception("Operands must be of the same dimension");

		add(vec);
		return *this;
	}

	Vector& Vector::operator -= (const Vector& vec)
	{
		if (vec.dim() != dim())
			throw std::exception("Operands must be of the same dimension");

		sub(vec);
		return *this;
	}

	Vector& Vector::operator *= (const Real& scalar)
	{
		mul(scalar);
		return *this;
	}

	Vector operator +(const Vector& vec1, const Vector& vec2)
	{
		auto result = vec1;
		return result += vec2;
	}

	Vector operator -(const Vector& vec1, const Vector& vec2)
	{
		auto result = vec1;
		return result -= vec2;
	}

	Vector operator *(const Vector& vec, const Real& scalar)
	{
		auto result = vec;
		return result *= scalar;
	}

	Vector operator *(const Real& scalar, const Vector& vec)
	{
		return vec * scalar;
	}

	void Vector::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		std::vector<Real> proxy;
		msgpack::type::make_define_array(proxy).msgpack_unpack(msgpack_o);
		assign(proxy);
	}

	void Vector::log(const std::filesystem::path& filename) const
	{
		Logging::log_as_table(get_handle(), dim(), 1, filename);
	}

	template Vector::Vector(const std::vector<unsigned char>& souurce);
	template Vector::Vector(const std::vector<Real>& souurce);
}