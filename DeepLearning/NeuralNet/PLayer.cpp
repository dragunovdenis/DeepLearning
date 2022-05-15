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

#include "PLayer.h"

namespace DeepLearning
{
	PLayer::PLayer(const Index3d& in_size, const Index2d& pool_window_size, const PoolTypeId pool_operator_id, const Index3d& strides)
		: _in_size(in_size), _pool_window_size(1, pool_window_size.x, pool_window_size.y), _strides(strides), _pool_operator_id(pool_operator_id)
	{
		if (_strides == Index3d{ 0 })
			_strides = _pool_window_size;
	}

	Index3d PLayer::in_size() const
	{
		return _in_size;
	}

	Index3d PLayer::out_size() const
	{
		return Tensor::calc_conv_res_size(_in_size, _pool_window_size, _paddings, _strides);
	}

	Index3d PLayer::weight_tensor_size() const
	{
		return _pool_window_size;
	}

	Tensor PLayer::act(const Tensor& input, AuxLearningData* const aux_learning_data_ptr) const
	{
		if (input.size_3d() != in_size())
			throw std::exception("Unexpected size of the input tensor");

		const auto pool_operator_ptr = PoolOperator::make(weight_tensor_size(), _pool_operator_id);

		if (aux_learning_data_ptr)
		{
			aux_learning_data_ptr->Input = input;
			aux_learning_data_ptr->Derivatives = Tensor({0});
		}

		return std::move(input.pool(*pool_operator_ptr, _paddings, _strides));
	}

	std::tuple<Tensor, PLayer::LayerGradient> PLayer::backpropagate(const Tensor& deltas, const AuxLearningData& aux_learning_data,
		const bool evaluate_input_gradient) const
	{
		if (deltas.size_3d() != out_size())
			throw std::exception("Unexpected size of the input tensor of derivatives");

		if (!evaluate_input_gradient)
			return std::make_tuple<Tensor, PLayer::LayerGradient>(Tensor(), { Tensor(), std::vector<Tensor>() });

		const auto pool_operator_ptr = PoolOperator::make(weight_tensor_size(), _pool_operator_id);
		auto input_grad = aux_learning_data.Input.pool_input_gradient(deltas, *pool_operator_ptr, _paddings, _strides);

		return std::make_tuple<Tensor, PLayer::LayerGradient>(std::move(input_grad), { Tensor(), std::vector<Tensor>() });
	}

	void PLayer::update(const std::tuple<std::vector<Tensor>, Tensor>& weights_and_biases_increment, const Real& reg_factor)
	{
		//Sanity check 
		if (std::get<0>(weights_and_biases_increment).size() != 0 || std::get<1>(weights_and_biases_increment).size() != 0)
			throw std::exception("There should be no increments for weights and/or biases");
	}

	CummulativeGradient PLayer::init_cumulative_gradient() const
	{
		return CummulativeGradient(0, 0);
	}
}