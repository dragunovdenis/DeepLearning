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

#include "Tensor.h"
#include "../Utilities.h"
#include <exception>
#include "../IndexIterator.h"

namespace DeepLearning
{
	Tensor::Tensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const bool assign_zero)
		: _layer_dim(layer_dim), _row_dim(row_dim), _col_dim(col_dim)
	{
		_data = reinterpret_cast<Real*>(std::malloc(size() * sizeof(Real)));

		if (assign_zero)
			std::fill(begin(), end(), Real(0));
	}

	Tensor::Tensor(const Tensor& tensor)
		: Tensor(tensor._layer_dim, tensor._row_dim, tensor._col_dim, false)
	{
		std::copy(tensor.begin(), tensor.end(), begin());
	}

	Tensor::Tensor(Tensor&& tensor) noexcept
		: _data(tensor._data), _layer_dim(tensor._layer_dim),
		_row_dim(tensor._row_dim), _col_dim(tensor._col_dim)
	{
		tensor._data = nullptr;
		tensor._layer_dim = 0;
		tensor._row_dim = 0;
		tensor._col_dim = 0;
	}

	Tensor::Tensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const Real range_begin, const Real range_end)
		: Tensor(layer_dim, row_dim, col_dim, false)
	{
		Utils::fill_with_random_values(begin(), end(), range_begin, range_end);
	}

	Tensor& Tensor::operator =(const Tensor& tensor)
	{
		if (size() != tensor.size())
		{
			free();
			_data = reinterpret_cast<Real*>(std::malloc(tensor.size() * sizeof(Real)));
		}

		_layer_dim = tensor._layer_dim;
		_row_dim = tensor._row_dim;
		_col_dim = tensor._col_dim;

		std::copy(tensor.begin(), tensor.end(), begin());

		return *this;
	}

	Tensor::~Tensor()
	{
		free();
	}

	void Tensor::free()
	{
		if (_data != nullptr)
		{
			delete[] _data;
			_data = nullptr;
		}

		_layer_dim = 0;
		_row_dim = 0;
		_col_dim = 0;
	}

	std::size_t Tensor::size() const
	{
		return _layer_dim * _row_dim * _col_dim;
	}

	Real* Tensor::begin()
	{
		return _data;
	}

	const Real* Tensor::begin() const
	{
		return _data;
	}

	Real* Tensor::end()
	{
		return _data + size();
	}

	const Real* Tensor::end() const
	{
		return _data + size();
	}

	std::size_t Tensor::layer_dim() const
	{
		return _layer_dim;
	}

	std::size_t Tensor::row_dim() const
	{
		return _row_dim;
	}

	std::size_t Tensor::col_dim() const
	{
		return _col_dim;
	}

	std::size_t Tensor::coords_to_data_id(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const
	{
		return _col_dim * (layer_id * _row_dim + row_id) + col_id;
	}

	bool Tensor::check_bounds(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const
	{
		return layer_id < _layer_dim && row_id < _row_dim && col_id < _col_dim;
	}

	Real& Tensor::operator ()(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, row_id, col_id))
			throw std::exception("Index out of bounds")
#endif // CHECK_BOUNDS

		return _data[coords_to_data_id(layer_id, row_id, col_id)];
	}

	const Real& Tensor::operator ()(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, row_id, col_id))
			throw std::exception("Index out of bounds")
#endif // CHECK_BOUNDS

		return _data[coords_to_data_id(layer_id, row_id, col_id)];
	}

	Tensor& Tensor::operator +=(const Tensor& tensor)
	{
		if (_layer_dim != tensor._layer_dim ||
			_row_dim != tensor._row_dim ||
			_col_dim != tensor._col_dim)
			throw std::exception("Invalid input");

		add(tensor);
		return *this;
	}

	Tensor& Tensor::operator -=(const Tensor& tensor)
	{
		if (_layer_dim != tensor._layer_dim ||
			_row_dim != tensor._row_dim ||
			_col_dim != tensor._col_dim)
			throw std::exception("Invalid input");

		sub(tensor);
		return *this;
	}

	Tensor& Tensor::operator *=(const Real& scalar)
	{
		mul(scalar);
		return *this;
	}

	Tensor operator +(const Tensor& tensor1, const Tensor& tensor2)
	{
		auto result = tensor1;
		result += tensor2;

		return result;
	}

	Tensor operator -(const Tensor& tensor1, const Tensor& tensor2)
	{
		auto result = tensor1;
		result -= tensor2;

		return result;
	}

	Tensor operator *(const Tensor& tensor, const Real& scalar)
	{
		auto result = tensor;
		result *= scalar;

		return result;
	}

	Tensor operator *(const Real& scalar, const Tensor& tensor)
	{
		return tensor * scalar;
	}

	bool Tensor::operator ==(const Tensor& tensor) const
	{
		return _layer_dim == tensor._layer_dim &&
			_row_dim == tensor._row_dim &&
			_col_dim == tensor._col_dim &&
			std::all_of(IndexIterator(0), IndexIterator(static_cast<int>(size())),
				[&](const auto& id) { return _data[id] == tensor._data[id]; });
	}

	bool Tensor::operator !=(const Tensor& tensor) const
	{
		return !(*this == tensor);
	}
}