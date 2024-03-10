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

#include "CummulativeGradient.h"
#include <algorithm>
#include <exception>
#include "DataContext.h"
#include "../Math/CollectionArithmetics.h"

namespace DeepLearning
{
	template <class D>
	CummulativeGradient<D>::CummulativeGradient(const Index3d& weight_tensor_size, const Index3d& bias_tensor_size)
	{
		const auto filters_cnt = bias_tensor_size.x;//Number of layers (channels) in the tensor of biases
		_gradient_sum.Weights_grad = std::vector<typename D::tensor_t>(filters_cnt, typename D::tensor_t(weight_tensor_size) );
		_gradient_sum.Biases_grad = typename D::tensor_t(bias_tensor_size);
	}

	template <class D>
	void CummulativeGradient<D>::add(const LayerGradient<D>& gradient)
	{
		_gradient_sum += gradient;
		_accumulated_items_count++;
	}

	template <class D>
	LayerGradient<D> CummulativeGradient<D>::calc_average_gradient(const Real scale_factor) const
	{
		if (_accumulated_items_count == 0)
			throw std::exception("No items have been added.");

		const auto factor = scale_factor / _accumulated_items_count;
		return _gradient_sum * factor;
	}

	template <class D>
	LayerGradient<D>& CummulativeGradient<D>::get_gradient_sum()
	{
		return _gradient_sum;
	}

	template <class D>
	std::size_t CummulativeGradient<D>::items_count() const
	{
		return _accumulated_items_count;
	}

	template <class D>
	void CummulativeGradient<D>::reset()
	{
		_gradient_sum.Biases_grad.fill_zero();
		for (auto item_id = 0ull; item_id < _gradient_sum.Weights_grad.size(); item_id++)
			_gradient_sum.Weights_grad[item_id].fill_zero();

		_accumulated_items_count = 0;
	}

	template class CummulativeGradient<CpuDC>;
	template class CummulativeGradient<GpuDC>;
}