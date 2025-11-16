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

namespace DeepLearning
{
	template<class D>
	typename D::matrix_t& RMLayer<D>::in_weights()
	{
		return _weights[IN_W];
	}

	template<class D>
	const typename D::matrix_t& RMLayer<D>::in_weights() const
	{
		return _weights[IN_W];
	}

	template<class D>
	typename D::matrix_t& RMLayer<D>::rec_weights()
	{
		return _weights[REC_W];
	}

	template<class D>
	const typename D::matrix_t& RMLayer<D>::rec_weights() const
	{
		return _weights[REC_W];
	}

	template<class D>
	long long& RMLayer<D>::rec_depth()
	{
		return _out_size.w;
	}

	template<class D>
	const long long& RMLayer<D>::rec_depth() const
	{
		return _out_size.w;
	}

	template<class D>
	Index3d& RMLayer<D>::out_sub_dim()
	{
		return _out_size.xyz;
	}

	template<class D>
	const Index3d& RMLayer<D>::out_sub_dim() const
	{
		return _out_size.xyz;
	}

	template <class D>
	const AFunction<typename D::tensor_t>& RMLayer<D>::get_func() const
	{
		return *_func;
	}

	template <class D>
	void RMLayer<D>::set_func_id(const ActivationFunctionId func_id)
	{
		_func_id = func_id;
		if (_func_id != ActivationFunctionId::UNKNOWN)
			_func = ActivationWrapper<typename D::tensor_t >::construct(_func_id);
		else
			_func.reset();
	}

	template <class D>
	void RMLayer<D>::apply_activation_function(typename D::tensor_t& in_out, typename D::tensor_t* const out_derivatives) const
	{
		if (out_derivatives)
		{
			get_func().func_and_aux_in_place(in_out, *out_derivatives);
		}
		else
			get_func().func_in_place(in_out);
	}

	template<class D>
	Index4d RMLayer<D>::in_size() const
	{
		return _in_size;
	}

	template<class D>
	Index4d RMLayer<D>::out_size() const
	{
		return _out_size;
	}

	template <class D>
	RMLayer<D>::RMLayer(const int rec_depth, const int in_sub_dim, const Index3d& out_sub_dim,
		const InitializationStrategy init_strategy, ActivationFunctionId func_id, Real gradient_clip_threshold,
		Real weight_scaling_factor) :
		RMLayer({{1, 1, in_sub_dim}, rec_depth }, { out_sub_dim, rec_depth }, init_strategy, func_id,
			gradient_clip_threshold, weight_scaling_factor) {}

	template <class D>
	RMLayer<D>::RMLayer(const Index4d& in_size, const Index4d& out_size, const InitializationStrategy init_strategy,
		ActivationFunctionId func_id, Real gradient_clip_threshold, Real weight_scaling_factor) :
		_in_size{ in_size }, _out_size{out_size}, _gradient_clip_threshold{ gradient_clip_threshold }
	{
		if (in_size.w != _out_size.w)
			throw std::exception("Unsupported input-output dimensionality");

		auto ran_gen_ptr = &AMLayer<D>::ran_gen();

		set_func_id(func_id);
		const auto out_sub_dim_lin = this->out_sub_dim().coord_prod();
		in_weights().resize(out_sub_dim_lin, _in_size.xyz.coord_prod());
		in_weights().init(init_strategy, ran_gen_ptr);

		rec_weights().resize(out_sub_dim_lin, out_sub_dim_lin);
		rec_weights().init(init_strategy, ran_gen_ptr);
		rec_weights() *= weight_scaling_factor;

		_biases.resize(out_sub_dim_lin);
		_biases.init(FillRandomUniform, ran_gen_ptr);
	}

	template<class D>
	MLayerGradient<D> RMLayer<D>::allocate_gradient_container(const bool fill_zero) const
	{
		MLayerGradient<D> result(1);

		auto& d = result[0].data;
		d.resize(_weights.size() + 1);
		d.shrink_to_fit();

		for (auto w_id = 1ull; w_id < d.size(); ++w_id)
			d[w_id].resize(_weights[w_id - 1].size_3d());

		d[BIAS_GRAD_ID].resize(_biases.size_3d());

		if (fill_zero)
			result.fill_zero();

		return result;
	}

	template<class D>
	MLayerData<D> RMLayer<D>::allocate_trace_data() const
	{
		return MLayerData<D>(std::max(rec_depth(), 0ll));
	}

	template <class D>
	void RMLayer<D>::act(const IMLayerExchangeData<typename D::tensor_t>& input,
		IMLayerExchangeData<typename D::tensor_t>& output, IMLayerTraceData<D>* const trace_data) const
	{
		const auto depth = rec_depth() <= 0 ? static_cast<long long>(input.size()) : rec_depth();

		if (depth != input.size() || (trace_data && trace_data->size() != depth))
			throw std::exception("Invalid input data");

		output.resize(depth);

		for (auto iter_id = 0; iter_id < depth; iter_id++)
		{
			output[iter_id].resize(out_sub_dim());

			in_weights().mul_add(input[iter_id], _biases, output[iter_id]);

			if (iter_id > 0)
				rec_weights().mul_add(output[iter_id - 1], output[iter_id], output[iter_id]);

			apply_activation_function(output[iter_id], trace_data ? &trace_data->trace(iter_id).Derivatives : nullptr);
		}
	}

	template <class D>
	void RMLayer<D>::backpropagate(const IMLayerExchangeData<typename D::tensor_t>& out_grad,
		const IMLayerExchangeData<typename D::tensor_t>& output, const IMLayerExchangeData<LayerData<D>>& processing_data,
		IMLayerExchangeData<typename D::tensor_t>& out_input_grad, MLayerGradient<D>& out_layer_grad,
		const bool evaluate_input_gradient) const
	{
		const auto depth = rec_depth() <= 0 ? static_cast<long long>(output.size()) : rec_depth();

		if (out_grad.size() != depth || output.size() != depth ||
			processing_data.size() != depth || out_layer_grad.size() != 1)
			throw std::exception("Invalid input.");

		if (evaluate_input_gradient) out_input_grad.resize(depth);

		thread_local typename D::tensor_t aux{};
		aux.resize(out_sub_dim());
		aux.fill_zero();

		thread_local typename D::tensor_t temp{};
		temp.resize(out_sub_dim());

		auto& biases_grad = out_layer_grad[0].data[BIAS_GRAD_ID];
		auto& in_weights_grad = out_layer_grad[0].data[IN_W_GRAD_ID];
		auto& rec_weights_grad = out_layer_grad[0].data[REC_W_GRAD_ID];

		for (auto step_id = depth - 1; step_id >= 0; --step_id)
		{
			get_func().add_in_grad(out_grad[step_id], processing_data[step_id].Trace.Derivatives, aux);
			biases_grad += aux;

			scale_and_add_vector_col_times_vector_row(aux, processing_data[step_id].Input,
				in_weights_grad, static_cast<Real>(1.0));

			if (evaluate_input_gradient)
				in_weights().transpose_mul(aux, out_input_grad[step_id]);

			if (step_id > 0)
			{
				scale_and_add_vector_col_times_vector_row(aux, output[step_id - 1],
					rec_weights_grad, static_cast<Real>(1.0));
				rec_weights().transpose_mul(aux, temp);
				get_func().calc_in_grad(temp, processing_data[step_id - 1].Trace.Derivatives, aux);
			}
		}
	}

	template <class D>
	void RMLayer<D>::update(const MLayerGradient<D>& increment, const Real learning_rate, const Real reg_factor)
	{
		// Apply gradient clipping if enabled
		Real effective_learning_rate = learning_rate;
		if (_gradient_clip_threshold > static_cast<Real>(0))
		{
			const Real grad_norm = increment.norm();
			if (grad_norm > _gradient_clip_threshold)
				effective_learning_rate *= _gradient_clip_threshold / grad_norm;
		}

		const auto& grad = increment[0];
		_biases.add_scaled(grad.data[BIAS_GRAD_ID], effective_learning_rate);

		if (reg_factor <= 0)
		{
			_weights[IN_W].add_scaled(grad.data[IN_W_GRAD_ID], effective_learning_rate);
			_weights[REC_W].add_scaled(grad.data[REC_W_GRAD_ID], effective_learning_rate);
		} else
		{
			const auto scale_factor = static_cast<Real>(1.0) + reg_factor * effective_learning_rate;
			_weights[IN_W].scale_and_add_scaled(scale_factor, grad.data[IN_W_GRAD_ID], effective_learning_rate);
			_weights[REC_W].scale_and_add_scaled(scale_factor, grad.data[REC_W_GRAD_ID], effective_learning_rate);
		}
	}

	template <class D>
	void RMLayer<D>::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		auto msg_pack_version = 0;
		msgpack::type::make_define_array(msg_pack_version).msgpack_unpack(msgpack_o);

		if (msg_pack_version != MSG_PACK_VER)
			throw std::exception("Unexpected version of an object");

		msgpack::type::make_define_array(msg_pack_version, MSGPACK_BASE(AMLayer<D>),
			_in_size, _out_size, _biases, _weights, _func_id, _gradient_clip_threshold).msgpack_unpack(msgpack_o);
		set_func_id(_func_id);
	}

	template <class D>
	template <typename Packer>
	void RMLayer<D>::msgpack_pack(Packer& msgpack_pk) const
	{
		msgpack::type::make_define_array(MSG_PACK_VER, MSGPACK_BASE(AMLayer<D>),
			_in_size, _out_size, _biases, _weights, _func_id, _gradient_clip_threshold).msgpack_pack(msgpack_pk);
	}

	template<class D>
	bool RMLayer<D>::equal_hyperparams(const AMLayer<D>& layer) const
	{
		const auto other_layer_ptr = dynamic_cast<const RMLayer*>(&layer);
		return other_layer_ptr != nullptr && AMLayer<D>::equal_hyperparams(layer)
			&& _in_size == other_layer_ptr->_in_size && _out_size == other_layer_ptr->_out_size &&
			_func_id == other_layer_ptr->_func_id &&
			_gradient_clip_threshold == other_layer_ptr->_gradient_clip_threshold;
	}

	template<class D>
	bool RMLayer<D>::equal(const AMLayer<D>& layer) const
	{
		if (!equal_hyperparams(layer))
			return false;

		const auto other_layer_ptr = dynamic_cast<const RMLayer*>(&layer);
		return other_layer_ptr != nullptr && AMLayer<D>::equal(layer)
			&& _biases == other_layer_ptr->_biases && _weights == other_layer_ptr->_weights;
	}

	template <class D>
	MLayerTypeId RMLayer<D>::get_type_id() const
	{
		return ID();
	}
}
