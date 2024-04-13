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
#include "../Math/ConvolutionUtils.h"
#include "../Utilities.h"
#include <nlohmann/json.hpp>

namespace DeepLearning
{
	template <class D>
	void PLayer<D>::initialize(const Index3d& in_size, const Index2d& pool_window_size, const PoolTypeId pool_operator_id)
	{
		_in_size = in_size;
		_pool_window_size = { 1, pool_window_size.x, pool_window_size.y };
		_pool_operator_id = pool_operator_id;
		_strides = _pool_window_size;
	}

	template <class D>
	void PLayer<D>::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		try
		{
			auto msg_pack_version = 0;
			msgpack::type::make_define_array(msg_pack_version, MSGPACK_BASE(ALayer<D>),
				_in_size, _pool_window_size, _strides, _pool_operator_id).msgpack_unpack(msgpack_o);
		}
		catch (...)
		{
			// to preserve backward compatibility
			Real keep_rate = -1;
			msgpack::type::make_define_array(keep_rate, _in_size, _pool_window_size, 
				_strides, _pool_operator_id).msgpack_unpack(msgpack_o);
			this->set_keep_rate(keep_rate);
		}
	}

	template <class D>
	PLayer<D>::PLayer(const Index3d& in_size, const Index2d& pool_window_size,
		const PoolTypeId pool_operator_id, const Real keep_rate) : ALayer<D>(keep_rate)
	{
		initialize(in_size, pool_window_size, pool_operator_id);
	}

	namespace {
		const char* json_in_size_id = "InSize";
		const char* json_pool_window_size_id = "FilterSize";
		const char* json_pool_operator_id = "PoolOperator";
	}

	template <class D>
	PLayer<D>::PLayer(const std::string& str, const Index3d& default_in_size) : ALayer<D>(str)
	{
		const auto json = nlohmann::json::parse(str);

		const auto in_size = json.contains(json_in_size_id) ?
			Utils::extract_vector<Index3d>(json[json_in_size_id].get<std::string>()) : default_in_size;

		const auto pool_window_size = json.contains(json_pool_window_size_id) ?
			Utils::extract_vector<Index2d>(json[json_pool_window_size_id].get<std::string>()) :
			throw std::exception("Can't parse window size of PLayer");

		const auto pool_operator_id = json.contains(json_pool_operator_id) ?
			parse_pool_type(json[json_pool_operator_id].get<std::string>()) :
			throw std::exception("Can't parse operator type of PLayer");

		initialize(in_size, pool_window_size, pool_operator_id);
	}

	template <class D>
	std::string PLayer<D>::to_script() const
	{
		nlohmann::json json = nlohmann::json::parse(ALayer<D>::to_script());

		json[json_in_size_id] = _in_size.to_string();
		json[json_pool_window_size_id] = _pool_window_size.yz().to_string();
		json[json_pool_operator_id] = DeepLearning::to_string(_pool_operator_id);

		return json.dump();
	}

	template <class D>
	Index3d PLayer<D>::in_size() const
	{
		return _in_size;
	}

	template <class D>
	Index3d PLayer<D>::out_size() const
	{
		return ConvolutionUtils::calc_conv_res_size(_in_size, _pool_window_size, {0}, _strides);
	}

	template <class D>
	Index3d PLayer<D>::weight_tensor_size() const
	{
		return _pool_window_size;
	}

	template <class D>
	void PLayer<D>::act(const typename D::tensor_t& input, typename D::tensor_t& output, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr) const
	{
		if (input.size_3d() != in_size())
			throw std::exception("Unexpected size of the input tensor");

		if (aux_learning_data_ptr)
		{
			aux_learning_data_ptr->Input = input;
			aux_learning_data_ptr->Derivatives.resize(Index3d(0));
		}

		if (_pool_operator_id == PoolTypeId::MIN || _pool_operator_id == PoolTypeId::MAX)
		{
			if (aux_learning_data_ptr)
				input.template min_max_pool<true>(_pool_window_size, _pool_operator_id == PoolTypeId::MAX, output, aux_learning_data_ptr->IndexMapping);
			else
				input.min_max_pool(_pool_window_size, _pool_operator_id == PoolTypeId::MAX, output);
		}
		else if (_pool_operator_id == PoolTypeId::AVERAGE)
			input.average_pool(_pool_window_size, output);
		else 
			throw std::exception("Unsupported pool type");
	}

	template <class D>
	typename D::tensor_t PLayer<D>::act(const typename D::tensor_t& input, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr) const
	{
		typename D::tensor_t result;
		act(input, result, aux_learning_data_ptr);

		return std::move(result);
	}

	template <class D>
	void PLayer<D>::backpropagate(const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
		typename D::tensor_t& input_grad, LayerGradient<D>& layer_grad, const bool evaluate_input_gradient,
		const Real gradient_scale_factor) const
	{
		if (deltas.size_3d() != out_size())
			throw std::exception("Unexpected size of the input tensor of derivatives");

		if (!evaluate_input_gradient)
		{
			input_grad.resize({ 0, 0, 0 });
			return;
		}

		if (_pool_operator_id == PoolTypeId::MIN || _pool_operator_id == PoolTypeId::MAX)
		{
			if (aux_learning_data.IndexMapping.size() != out_size().coord_prod())
				throw std::exception("Invalid index mapping");

			aux_learning_data.Input.min_max_pool_input_gradient(deltas, aux_learning_data.IndexMapping, input_grad);
			return;
		}

		if (_pool_operator_id != PoolTypeId::AVERAGE)
			throw std::exception("Unsupported pool type");

		aux_learning_data.Input.average_pool_input_gradient(deltas, _pool_window_size, input_grad);
	}

	template <class D>
	void PLayer<D>::allocate(LayerGradient<D>& gradient_container, bool fill_zeros) const
	{
		gradient_container.Biases_grad.resize({ 0, 0, 0 });
		gradient_container.Weights_grad.resize(0);
	}

	template <class D>
	std::tuple<typename D::tensor_t, LayerGradient<D>> PLayer<D>::backpropagate(
		const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
		const bool evaluate_input_gradient) const
	{
		typename D::tensor_t input_grad;
		LayerGradient<D> layer_grad;
		allocate(layer_grad, /*fill zeros*/ false);
		backpropagate(deltas, aux_learning_data, input_grad, layer_grad, evaluate_input_gradient);
		return std::make_tuple(std::move(input_grad), std::move(layer_grad));
	}

	template <class D>
	void PLayer<D>::update(const LayerGradient<D>& gradient, const Real& l_rate, const Real& reg_factor)
	{
		//Sanity check 
		if (gradient.Weights_grad.size() != 0 || gradient.Biases_grad.size() != 0)
			throw std::exception("There should be no increments for weights and/or biases");
	}

	template <class D>
	CummulativeGradient<D> PLayer<D>::init_cumulative_gradient() const
	{
		return CummulativeGradient<D>(0, 0);
	}

	template <class D>
	std::string PLayer<D>::to_string() const
	{
		return DeepLearning::to_string(PLayer::ID()) + "; " + to_script();
	}

	template <class D>
	bool PLayer<D>::equal_hyperparams(const ALayer<D>& layer) const
	{
		const auto other_player_ptr = dynamic_cast<const PLayer*>(&layer);
		return other_player_ptr != nullptr && 
			ALayer<D>::equal_hyperparams(layer) && _in_size == layer.in_size() &&
			_pool_window_size == layer.weight_tensor_size() &&
			_strides == other_player_ptr->_strides &&
			_pool_operator_id == other_player_ptr->_pool_operator_id;
	}

	template <class D>
	bool PLayer<D>::equal(const ALayer<D>& layer) const
	{
		return equal_hyperparams(layer); // no other parameters to check for equality
	}

	template <class D>
	LayerTypeId PLayer<D>::get_type_id() const
	{
		return ID();
	}

	template <class D>
	Real PLayer<D>::squared_weights_sum() const
	{
		return Real(0);
	}

	template <>
	PLayer<CpuDC> PLayer<CpuDC>::to_host() const
	{
		return *this;
	}

	template <>
	PLayer<CpuDC> PLayer<GpuDC>::to_host() const
	{
		return PLayer<CpuDC>(_in_size, _pool_window_size.yz(), _pool_operator_id);
	}

	template <>
	PLayer<GpuDC> PLayer<GpuDC>::to_device() const
	{
		return *this;
	}

	template <class D>
	void PLayer<D>::reset()
	{
		//no "trainable" parameters so nothing to reset
	}

	template <>
	PLayer<GpuDC> PLayer<CpuDC>::to_device() const
	{
		return PLayer<GpuDC>(_in_size, _pool_window_size.yz(), _pool_operator_id);
	}

	template class PLayer<CpuDC>;
	template class PLayer<GpuDC>;
}