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

#include "DenseTensor.h"
#include "../Utilities.h"
#include <exception>
#include "../IndexIterator.h"

namespace DeepLearning
{
	DenseTensor::DenseTensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const bool assign_zero)
		: _layer_dim(layer_dim), _row_dim(row_dim), _col_dim(col_dim)
	{
		_data = reinterpret_cast<Real*>(std::malloc(size() * sizeof(Real)));

		if (assign_zero)
			std::fill(begin(), end(), Real(0));
	}

	DenseTensor::DenseTensor(const DenseTensor& tensor)
		: DenseTensor(tensor._layer_dim, tensor._row_dim, tensor._col_dim, false)
	{
		std::copy(tensor.begin(), tensor.end(), begin());
	}

	DenseTensor::DenseTensor(DenseTensor&& tensor) noexcept
		: _data(tensor._data), _layer_dim(tensor._layer_dim),
		_row_dim(tensor._row_dim), _col_dim(tensor._col_dim)
	{
		tensor._data = nullptr;
		tensor._layer_dim = 0;
		tensor._row_dim = 0;
		tensor._col_dim = 0;
	}

	DenseTensor::DenseTensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const Real range_begin, const Real range_end)
		: DenseTensor(layer_dim, row_dim, col_dim, false)
	{
		Utils::fill_with_random_values(begin(), end(), range_begin, range_end);
	}

	DenseTensor& DenseTensor::operator =(const DenseTensor& tensor)
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

	DenseTensor::~DenseTensor()
	{
		free();
	}

	void DenseTensor::free()
	{
		if (_data != nullptr)
			delete[] _data;

		_layer_dim = 0;
		_row_dim = 0;
		_col_dim = 0;
	}

	std::size_t DenseTensor::size() const
	{
		return _layer_dim * _row_dim * _col_dim;
	}

	Real* DenseTensor::begin()
	{
		return _data;
	}

	const Real* DenseTensor::begin() const
	{
		return _data;
	}

	Real* DenseTensor::end()
	{
		return _data + size();
	}

	const Real* DenseTensor::end() const
	{
		return _data + size();
	}

	std::size_t DenseTensor::layer_dim() const
	{
		return _layer_dim;
	}

	std::size_t DenseTensor::row_dim() const
	{
		return _row_dim;
	}

	std::size_t DenseTensor::col_dim() const
	{
		return _col_dim;
	}

	DenseTensor& DenseTensor::operator +=(const DenseTensor& tensor)
	{
		if (_layer_dim != tensor._layer_dim ||
			_row_dim != tensor._row_dim ||
			_col_dim != tensor._col_dim)
			throw std::exception("Invalid input");

		add(tensor);
		return *this;
	}

	DenseTensor& DenseTensor::operator -=(const DenseTensor& tensor)
	{
		if (_layer_dim != tensor._layer_dim ||
			_row_dim != tensor._row_dim ||
			_col_dim != tensor._col_dim)
			throw std::exception("Invalid input");

		sub(tensor);
		return *this;
	}

	DenseTensor& DenseTensor::operator *=(const Real& scalar)
	{
		mul(scalar);
		return *this;
	}

	DenseTensor operator +(const DenseTensor& tensor1, const DenseTensor& tensor2)
	{
		auto result = tensor1;
		result += tensor2;

		return result;
	}

	DenseTensor operator -(const DenseTensor& tensor1, const DenseTensor& tensor2)
	{
		auto result = tensor1;
		result -= tensor2;

		return result;
	}

	DenseTensor operator *(const DenseTensor& tensor, const Real& scalar)
	{
		auto result = tensor;
		result *= scalar;

		return result;
	}

	DenseTensor operator *(const Real& scalar, const DenseTensor& tensor)
	{
		return tensor * scalar;
	}

	bool DenseTensor::operator ==(const DenseTensor& tensor) const
	{
		return _layer_dim == tensor._layer_dim &&
			_row_dim == tensor._row_dim &&
			_col_dim == tensor._col_dim &&
			std::all_of(IndexIterator(0), IndexIterator(static_cast<int>(size())),
				[&](const auto& id) { return _data[id] == tensor._data[id]; });
	}

	bool DenseTensor::operator !=(const DenseTensor& tensor) const
	{
		return !(*this == tensor);
	}
}