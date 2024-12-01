//Copyright (c) 2024 Denys Dragunov, dragunovdenis@gmail.com
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

#pragma once
#include <vector>
#include "IMLayerExchangeData.h"

namespace DeepLearning
{
	/// <summary>
	/// A collection that does not call destructors of its items when shrinking.
	/// </summary>
	template <class T>
	class LazyVector : public IMLayerExchangeData<T>
	{
		std::vector<T> _data{};
		std::size_t _size{0};

	public:

		/// <summary>
		/// Returns current size.
		/// </summary>
		std::size_t size() const override
		{
			return _size;
		}

		/// <summary>
		/// Resizes the collection in a lazy manner,i.e., does not do anything if
		/// the underlying std::vector has size greater or equal to the given requested size.
		/// </summary>
		void resize(const std::size_t new_size) override
		{
			_size = new_size;

			if (new_size <= _data.size())
				return;

			_data.resize(new_size);
		}

		/// <summary>
		/// Actually clears the underlying collection of items.
		/// </summary>
		void reset()
		{
			_data.clear();
			_size = 0;
		}

		/// <summary>
		/// Subscript operator.
		/// </summary>
		T& operator [](const std::size_t item_id) override
		{
			return _data[item_id];
		}

		/// <summary>
		/// Subscript operator ("const" version).
		/// </summary>
		const T& operator [](const std::size_t item_id) const override
		{
			return _data[item_id];
		}

		/// <summary>
		/// Maximal absolute 
		/// </summary>
		Real max_abs() const
		{
			auto result = static_cast<Real>(0);

			for (auto item_id = 0ull; item_id < size(); ++item_id)
				result = std::max(result, _data[item_id].max_abs());

			return result;
		}

		/// <summary>
		/// Sum of all the items. 
		/// </summary>
		Real sum() const
		{
			auto result = static_cast<Real>(0);

			for (auto item_id = 0ull; item_id < size(); ++item_id)
				result += _data[item_id].sum();

			return result;
		}

		/// <summary>
		/// Default constructor.
		/// </summary>
		LazyVector() = default;

		/// <summary>
		/// Constructor.
		/// </summary>
		LazyVector(const std::size_t size) : _data(size), _size(size){}

		/// <summary>
		/// Compound addition operator.
		/// </summary>
		LazyVector& operator +=(const LazyVector& v)
		{
			if (size() != v.size())
				throw std::exception("Invalid input");

			for (auto item_id = 0ull; item_id < size(); item_id++)
				_data[item_id] += v[item_id];

			return *this;
		}

		/// <summary>
		/// Compound subtraction operator.
		/// </summary>
		LazyVector& operator -=(const LazyVector& v)
		{
			if (size() != v.size())
				throw std::exception("Invalid input");

			for (auto item_id = 0ull; item_id < size(); item_id++)
				_data[item_id] -= v[item_id];

			return *this;
		}

		/// <summary>
		/// Compound multiplication by scalar operator.
		/// </summary>
		LazyVector& operator *=(const Real scalar)
		{
			for (auto item_id = 0ull; item_id < size(); item_id++)
				_data[item_id] *= scalar;

			return *this;
		}

		/// <summary>
		/// Compound division by scalar operator.
		/// </summary>
		LazyVector& operator /=(const Real scalar)
		{
			const auto one_over_scalar = static_cast<Real>(1) / scalar;
			return (*this) *= one_over_scalar;
		}
	};

	/// <summary>
	/// Addition operator.
	/// </summary>
	template <class T>
	LazyVector<T> operator +(const LazyVector<T>& v0, const LazyVector<T>& v1)
	{
		auto result = v0;
		return result += v1;
	}

	/// <summary>
	/// Subtraction operator.
	/// </summary>
	template <class T>
	LazyVector<T> operator -(const LazyVector<T>& v0, const LazyVector<T>& v1)
	{
		auto result = v0;
		return result -= v1;
	}

	/// <summary>
	/// Multiplication by a scalar from the right operator.
	/// </summary>
	template <class T>
	LazyVector<T> operator *(const LazyVector<T>& v, const Real scalar)
	{
		auto result = v;
		return result *= scalar;
	}

	/// <summary>
	/// Multiplication by a scalar from the left operator.
	/// </summary>
	template <class T>
	LazyVector<T> operator *(const Real scalar, const LazyVector<T>& v)
	{
		auto result = v;
		return result *= scalar;
	}
}
