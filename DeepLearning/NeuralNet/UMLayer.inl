//Copyright (c) 2026 Denys Dragunov, dragunovdenis@gmail.com
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
#include "UMLayer.h"

namespace DeepLearning
{
	template <class D, template<class> class L>
	template <typename... Args>
	UMLayer<D, L>::UMLayer(const long long depth, Args... core_layer_args):
		_core(core_layer_args...), _depth(depth)
	{}

	template <class D, template<class> class L>
	ALayer<D>& UMLayer<D, L>::core_layer()
	{
		return static_cast<ALayer<D>&>(_core);
	}

	template <class D, template<class> class L>
	const ALayer<D>& UMLayer<D, L>::core_layer() const
	{
		return static_cast<const ALayer<D>&>(_core);
	}

	template <class D, template<class> class L>
	Index4d UMLayer<D, L>::in_size() const
	{
		return { {_core.in_size()}, _depth };
	}

	template <class D, template<class> class L>
	Index4d UMLayer<D, L>::out_size() const
	{
		return { {_core.out_size()}, _depth };
	}

	template <class D, template<class> class L>
	MLayerGradient<D> UMLayer<D, L>::allocate_gradient_container(const bool fill_zero) const
	{
		MLayerGradient<D> gradient_container(1);
		_core.allocate(gradient_container[0], fill_zero);

		return gradient_container;
	}

	template <class D, template<class> class L>
	MLayerData<D> UMLayer<D, L>::allocate_trace_data() const
	{
		return MLayerData<D>(0);
	}

	template <class D, template<class> class L>
	void UMLayer<D, L>::update(const MLayerGradient<D>& increment, const Real learning_rate, const Real reg_factor)
	{
		const auto check = increment.norm();

		_core.update(increment[0], learning_rate, reg_factor);
	}

	template <class D, template<class> class L>
	void UMLayer<D, L>::act(const IMLayerExchangeData<typename D::tensor_t>& input,
		IMLayerExchangeData<typename D::tensor_t>& output, IMLayerTraceData<D>* const trace_data) const
	{
		const auto depth = _depth <= 0 ? static_cast<long long>(input.size()) : _depth;

		if (depth != input.size() || (trace_data && trace_data->size() != depth))
			throw std::exception("Invalid input data");

		output.resize(depth);

		const auto out_sub_dim = _core.out_size();

		for (auto i = 0; i < depth; i++)
		{
			output[i].resize(out_sub_dim);
			core_layer().act(input[i], output[i], trace_data ? &trace_data->trace(i) : nullptr);
		}
	}

	template <class D, template<class> class L>
	void UMLayer<D, L>::backpropagate(const IMLayerExchangeData<typename D::tensor_t>& out_grad,
		const IMLayerExchangeData<typename D::tensor_t>& output,
		const IMLayerExchangeData<LayerData<D>>& processing_data,
		IMLayerExchangeData<typename D::tensor_t>& out_input_grad,
		MLayerGradient<D>& out_layer_grad, const bool evaluate_input_gradient) const
	{
		const auto depth = _depth <= 0 ? static_cast<long long>(output.size()) : _depth;

		if (out_grad.size() != depth || output.size() != depth ||
			processing_data.size() != depth || out_layer_grad.size() != 1)
			throw std::exception("Invalid input.");

		out_input_grad.resize(depth);

		auto& layer_grad = out_layer_grad[0];

		for (auto i = 0; i < depth; i++)
		{
			core_layer().backpropagate(out_grad[i], processing_data[i], 
				out_input_grad[i], layer_grad, evaluate_input_gradient, static_cast<Real>(1));
		}
	}

	template <class D, template<class> class L>
	bool UMLayer<D, L>::equal_hyperparams(const AMLayer<D>& layer) const
	{
		const auto other_layer_ptr = dynamic_cast<const UMLayer*>(&layer);
		return other_layer_ptr != nullptr && AMLayer<D>::equal_hyperparams(layer) &&
			core_layer().equal_hyperparams(other_layer_ptr->_core);
	}

	template <class D, template<class> class L>
	bool UMLayer<D, L>::equal(const AMLayer<D>& layer) const
	{
		if (!equal_hyperparams(layer))
			return false;

		const auto other_layer_ptr = dynamic_cast<const UMLayer*>(&layer);
		return other_layer_ptr != nullptr && AMLayer<D>::equal(layer) &&
			core_layer().equal(other_layer_ptr->_core);
	}

	template <class D, template<class> class L>
	void UMLayer<D, L>::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		auto msg_pack_version = 0;
		msgpack::type::make_define_array(msg_pack_version).msgpack_unpack(msgpack_o);

		if (msg_pack_version != MSG_PACK_VER)
			throw std::exception("Unexpected version of an object");

		msgpack::type::make_define_array(msg_pack_version, MSGPACK_BASE(AMLayer<D>),
			_core).msgpack_unpack(msgpack_o);
	}

	template <class D, template<class> class L>
	template <typename Packer>
	void UMLayer<D, L>::msgpack_pack(Packer& msgpack_pk) const
	{
		msgpack::type::make_define_array(MSG_PACK_VER,
			MSGPACK_BASE(AMLayer<D>), _core).msgpack_pack(msgpack_pk);
	}

	template <class D>
	MLayerTypeId UNMLayer<D>::get_type_id() const
	{
		return ID();
	}
}
