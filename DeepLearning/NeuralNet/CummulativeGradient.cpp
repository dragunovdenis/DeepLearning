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

namespace DeepLearning
{
	CummulativeGradient::CummulativeGradient(const std::size_t in_dim, const std::size_t out_dim)
	{
		_sum_grad_weights = { Matrix(out_dim, in_dim) };
		_sum_grad_biases = Vector(out_dim);
	}

	void CummulativeGradient::Add(const std::vector<Tensor>& weight_grad, const Tensor& bias_grad)
	{
		if (_sum_grad_weights.size() != weight_grad.size())
			throw std::exception("Invalid input");

		for (auto item_id = 0ull; item_id < _sum_grad_weights.size(); item_id++)
			_sum_grad_weights[item_id] += weight_grad[item_id];

		_sum_grad_biases += bias_grad;
		_accumulated_items_count++;
	}

	std::tuple<std::vector<Tensor>, Tensor> CummulativeGradient::calc_average_grarient(const Real scale_factor) const
	{
		if (_accumulated_items_count == 0)
			throw std::exception("No items have been added.");

		const auto factor = scale_factor / _accumulated_items_count;
		auto average_grad_weights = _sum_grad_weights;

		for (auto item_id = 0ull; item_id < _sum_grad_weights.size(); item_id++)
			average_grad_weights[item_id] *= factor;

		return std::make_tuple(average_grad_weights, _sum_grad_biases * factor);
	}

	void CummulativeGradient::reset()
	{
		_sum_grad_biases.fill(Real(0));
		for (auto item_id = 0ull; item_id < _sum_grad_weights.size(); item_id++)
			_sum_grad_weights[item_id].fill(Real(0));

		_accumulated_items_count = 0;
	}
}