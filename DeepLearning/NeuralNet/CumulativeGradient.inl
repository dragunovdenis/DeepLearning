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

#pragma once

#include <exception>

namespace DeepLearning
{
	template <class D>
	CumulativeGradient<D>::CumulativeGradient(LayerGradient<D>&& seed) : _gradient_sum{std::move(seed)} {}

	template <class D>
	void CumulativeGradient<D>::add(const LayerGradient<D>& gradient)
	{
		_gradient_sum += gradient;
		_accumulated_items_count++;
	}

	template <class D>
	LayerGradient<D> CumulativeGradient<D>::calc_average_gradient(const Real scale_factor) const
	{
		if (_accumulated_items_count == 0)
			throw std::exception("No items have been added.");

		const auto factor = scale_factor / _accumulated_items_count;
		return _gradient_sum * factor;
	}

	template <class D>
	LayerGradient<D>& CumulativeGradient<D>::get_gradient_sum()
	{
		return _gradient_sum;
	}

	template <class D>
	std::size_t CumulativeGradient<D>::items_count() const
	{
		return _accumulated_items_count;
	}

	template <class D>
	void CumulativeGradient<D>::reset()
	{
		for (auto& item : _gradient_sum.data)
			item.fill_zero();

		_accumulated_items_count = 0;
	}
}