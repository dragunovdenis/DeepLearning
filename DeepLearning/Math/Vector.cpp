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
#include <algorithm>
#include <numeric>
#include "../IndexIterator.h"
#include "..//Diagnostics/Logging.h"

namespace DeepLearning
{
	Vector::Vector(const Vector& vec)
		: Vector(vec.dim(), false)
	{
		std::ranges::copy(vec, begin());
	}

	Vector& Vector::operator=(const Vector& vec)
	{
		if (this != &vec)
			assign(vec);

		return *this;
	}

	Vector& Vector::operator=(Vector&& vec) noexcept
	{
		if (this != &vec)
		{
			_dim = vec._dim;
			take_over_resources(std::move(vec));
		}

		return *this;
	}

	void Vector::abandon_resources()
	{
		Base::abandon_resources();
		_dim = 0;
	}

	Vector::Vector(Vector&& vec) noexcept : _dim(vec._dim)
	{
		take_over_resources(std::move(vec));
	}

	Vector::Vector(const std::size_t dim, const bool assign_zero)
	{
		resize(dim);

		if (assign_zero)
			fill_zero();
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
		std::ranges::copy(source, begin());
	}

	Vector::Vector(const std::size_t dim, const Real range_begin, const Real range_end, std::mt19937* seeder)
		: Vector(dim, false)
	{
		uniform_random_fill(range_begin, range_end, seeder);
	}

	template <class S>
	void Vector::assign(const S& source)
	{
		resize(source.size());
		std::ranges::copy(source, begin());
	}

	template void Vector::assign(const std::vector<Real>& source);
	template void Vector::assign(const Vector& source);

	void Vector::resize(const std::size_t& new_size)
	{
		allocate(new_size);
		_dim = new_size;
	}

	void Vector::resize(const Index3d& size_3d)
	{
		if (size_3d.x != 1ll || size_3d.y != 1ll)
			throw std::exception("Invalid input data");

		resize(size_3d.z);
	}

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

	bool Vector::operator ==(const Vector& vec) const
	{
		const auto data = begin();
		const auto vec_data = vec.begin();
		return _dim == vec._dim && std::all_of(IndexIterator(0), IndexIterator(static_cast<int>(_dim)),
			[&](const auto& index) { return data[index] == vec_data[index]; });
	}

	bool Vector::operator !=(const Vector& vec) const
	{
		return !(*this == vec);
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

	void Vector::fill_with_random_selection_map(const std::size_t& selected_cnt,
		std::vector<int>& aux_collection, std::mt19937* seeder)
	{
		if (selected_cnt >= size())
		{
			fill(static_cast<Real>(1));
			return;
		}

		aux_collection.resize(size());
		std::iota(aux_collection.begin(), aux_collection.end(), 0);
		uniform_random_fill(Real(-1), Real(1), seeder);

		std::ranges::sort(aux_collection, [this](const auto a, const auto b)
		{
			return this->begin()[a] < this->begin()[b];
		});

		auto data = begin();

		for (auto id = 0ull; id < selected_cnt; id++)
			data[aux_collection[id]] = Real(1);

		for (auto id = selected_cnt; id < size(); id++)
			data[aux_collection[id]] = Real(0);
	}

	void Vector::fill_with_random_selection_map(const std::size_t& selected_cnt)
	{
		std::vector<int> aux_collection;
		fill_with_random_selection_map(selected_cnt, aux_collection);
	}

	template Vector::Vector(const std::vector<unsigned char>& source);
	template Vector::Vector(const std::vector<Real>& source);
}