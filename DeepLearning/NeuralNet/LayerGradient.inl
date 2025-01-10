//Copyright (c) 2023 Denys Dragunov, dragunovdenis@gmail.com
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
#include "../Math/CollectionArithmetics.h"

namespace DeepLearning
{
	template <class D>
	bool LayerGradient<D>::operator ==(const LayerGradient& lg) const
	{
		return data == lg.data;
	}

	template <class D>
	bool LayerGradient<D>::operator !=(const LayerGradient& lg) const
	{
		return !(*this == lg);
	}

	template <class D>
	LayerGradient<D>& LayerGradient<D>::operator +=(const LayerGradient& lg)
	{
		data += lg.data;
		return *this;
	}

	template <class D>
	LayerGradient<D> operator +(const LayerGradient<D>& lg, const LayerGradient<D>& lg1)
	{
		auto result = lg;
		return result += lg1;
	}

	template <class D>
	LayerGradient<D>& LayerGradient<D>::operator -=(const LayerGradient& lg)
	{
		data -= lg.data;
		return *this;
	}

	template <class D>
	LayerGradient<D> operator -(const LayerGradient<D>& lg, const LayerGradient<D>& lg1)
	{
		auto result = lg;
		return result -= lg1;
	}

	template <class D>
	LayerGradient<D>& LayerGradient<D>::operator *=(const Real& scalar)
	{
		data *= scalar;
		return *this;
	}

	template <class D>
	LayerGradient<D> operator *(const LayerGradient<D>& lg, const Real& scalar)
	{
		auto result = lg;
		return result *= scalar;
	}

	template <class D>
	LayerGradient<D> operator *(const Real& scalar, const LayerGradient<D>& lg)
	{
		return lg * scalar;
	}

	template <class D>
	Real LayerGradient<D>::max_abs() const
	{
		auto result = static_cast<Real>(0);

		for (const auto& item : data)
			result = std::max(result, item.max_abs());

		return result;
	}

	template <class D>
	bool LayerGradient<D>::empty() const
	{
		return data.empty();
	}

	template <class D>
	LayerGradient<D>& LayerGradient<D>::add_scaled(const LayerGradient& lg, const Real& scalar)
	{
		if (data.size() != lg.data.size())
			throw std::exception("Inconsistent data");

		for (auto item_id = 0ull; item_id < data.size(); ++item_id)
			data[item_id].add_scaled(lg.data[item_id], scalar);

		return *this;
	}

	template <class D>
	LayerGradient<D>& LayerGradient<D>::scale_and_add(const Real& scalar, const LayerGradient& lg)
	{
		if (data.size() != lg.data.size())
			throw std::exception("Inconsistent data");

		for (auto item_id = 0ull; item_id < data.size(); ++item_id)
			data[item_id].scale_and_add(lg.data[item_id], scalar);

		return *this;
	}

	template<class D>
	void LayerGradient<D>::fill_zero()
	{
		for (auto& w : data)
			w.fill_zero();
	}
}