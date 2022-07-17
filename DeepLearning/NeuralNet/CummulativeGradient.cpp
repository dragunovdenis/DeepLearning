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

namespace DeepLearning
{
	template <class D>
	CummulativeGradient<D>::CummulativeGradient(const Index3d& weight_tensor_size, const Index3d& bias_tensor_size)
	{
		const auto filters_cnt = bias_tensor_size.x;//Number of layers (channels) in the tensor of biases
		_sum_grad_weights = std::vector<typename D::tensor_t>(filters_cnt, typename D::tensor_t(weight_tensor_size) );
		_sum_grad_biases = typename D::tensor_t(bias_tensor_size);
	}

	template <class D>
	void CummulativeGradient<D>::add(const std::vector<typename D::tensor_t>& weight_grad, const typename D::tensor_t& bias_grad)
	{
		if (weight_grad.size() != 0)
			_sum_grad_weights += weight_grad;

		if (bias_grad.size() != 0)
			_sum_grad_biases += bias_grad;

		_accumulated_items_count++;
	}

	template <class D>
	void CummulativeGradient<D>::add(const CummulativeGradient& gradient)
	{
		if (gradient._sum_grad_weights.size() != 0)
			_sum_grad_weights += gradient._sum_grad_weights;

		if (gradient._sum_grad_biases.size() != 0)
			_sum_grad_biases += gradient._sum_grad_biases;

		_accumulated_items_count += gradient._accumulated_items_count;
	}

	template <class D>
	std::tuple<std::vector<typename D::tensor_t>, typename D::tensor_t> CummulativeGradient<D>::calc_average_grarient(const Real scale_factor) const
	{
		if (_accumulated_items_count == 0)
			throw std::exception("No items have been added.");

		const auto factor = scale_factor / _accumulated_items_count;
		auto average_grad_weights = _sum_grad_weights;

		average_grad_weights *= factor;

		return std::make_tuple(average_grad_weights, _sum_grad_biases * factor);
	}

	template <class D>
	void CummulativeGradient<D>::reset()
	{
		_sum_grad_biases.fill(Real(0));
		for (auto item_id = 0ull; item_id < _sum_grad_weights.size(); item_id++)
			_sum_grad_weights[item_id].fill(Real(0));

		_accumulated_items_count = 0;
	}

	template class CummulativeGradient<CpuDC>;
	template class CummulativeGradient<GpuDC>;
}