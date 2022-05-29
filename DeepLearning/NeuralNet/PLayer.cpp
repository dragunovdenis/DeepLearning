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
	void PLayer::initialize(const Index3d& in_size, const Index2d& pool_window_size, const PoolTypeId pool_operator_id)
	{
		_in_size = in_size;
		_pool_window_size = { 1, pool_window_size.x, pool_window_size.y };
		_pool_operator_id = pool_operator_id;
		_strides = _pool_window_size;
	}

	PLayer::PLayer(const Index3d& in_size, const Index2d& pool_window_size, const PoolTypeId pool_operator_id)
	{
		initialize(in_size, pool_window_size, pool_operator_id);
	}

	PLayer::PLayer(const std::string& str)
	{
		auto str_norm = Utils::normalize_string(str);

		Index3d temp_3d;
		if (!Utils::try_extract_vector(str_norm, temp_3d))
			throw std::exception("Can't parse input dimensions of PLayer");

		const auto in_size = temp_3d;

		Index2d temp_2d;
		if (!Utils::try_extract_vector(str_norm, temp_2d))
			throw std::exception("Can't parse input dimensions of PLayer");

		const auto pool_window_size = temp_2d;

		const auto pool_operator_id = parse_pool_type(Utils::extract_word(str_norm));

		if (pool_operator_id == PoolTypeId::UNKNOWN)
			throw std::exception("Failed to parse pool operator type");

		initialize(in_size, pool_window_size, pool_operator_id);
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

		if (aux_learning_data_ptr)
		{
			aux_learning_data_ptr->Input = input;
			aux_learning_data_ptr->Derivatives = Tensor({0});
		}

		if (_pool_operator_id == PoolTypeId::MIN || _pool_operator_id == PoolTypeId::MAX)
		{
			auto [pool_result, index_mapping] = input.min_max_pool_2d({ _pool_window_size.y, _pool_window_size.z }, _pool_operator_id == PoolTypeId::MAX);

			if (aux_learning_data_ptr)
				aux_learning_data_ptr->IndexMapping = std::move(index_mapping);

			return std::move(pool_result);
		}

		const auto pool_operator_ptr = PoolOperator::make(weight_tensor_size(), _pool_operator_id);
		return std::move(input.pool(*pool_operator_ptr, _paddings, _strides));
	}

	std::tuple<Tensor, PLayer::LayerGradient> PLayer::backpropagate(const Tensor& deltas, const AuxLearningData& aux_learning_data,
		const bool evaluate_input_gradient) const
	{
		if (deltas.size_3d() != out_size())
			throw std::exception("Unexpected size of the input tensor of derivatives");

		if (!evaluate_input_gradient)
			return std::make_tuple<Tensor, PLayer::LayerGradient>(Tensor(), { Tensor(), std::vector<Tensor>() });

		if (_pool_operator_id == PoolTypeId::MIN || _pool_operator_id == PoolTypeId::MAX)
		{
			if (aux_learning_data.IndexMapping.size() != out_size().coord_prod())
				throw std::exception("Invalid index mapping");

			auto input_grad = aux_learning_data.Input.min_max_pool_2d_input_gradient(deltas, aux_learning_data.IndexMapping);
			return std::make_tuple<Tensor, PLayer::LayerGradient>(std::move(input_grad), { Tensor(), std::vector<Tensor>() });
		}

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

	std::string PLayer::to_string() const
	{
		return DeepLearning::to_string(PLayer::ID()) + "; Input size: " + in_size().to_string() +
			"; Out size: " + out_size().to_string() + "; Filter size: " + weight_tensor_size().to_string() +
			"; Pool type: " + DeepLearning::to_string(_pool_operator_id);
	}
}