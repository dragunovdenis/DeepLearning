#include "RMLayer.h"
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
	inline typename D::matrix_t& RMLayer<D>::in_weights()
	{
		return _weights[IN_W];
	}

	template<class D>
	inline const typename D::matrix_t& RMLayer<D>::in_weights() const
	{
		return _weights[IN_W];
	}

	template<class D>
	inline typename D::matrix_t& RMLayer<D>::rec_weights()
	{
		return _weights[REC_W];
	}

	template<class D>
	inline const typename D::matrix_t& RMLayer<D>::rec_weights() const
	{
		return _weights[REC_W];
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

	template <class D>
	RMLayer<D>::RMLayer(const int rec_depth, const int in_sub_dim, const Index3d& out_sub_dim,
		const InitializationStrategy init_strategy, ActivationFunctionId func_id) :
		_rec_depth(rec_depth), _out_sub_dim(out_sub_dim)
	{
		set_func_id(func_id);
		const auto out_sub_dim_lin = _out_sub_dim.coord_prod();
		in_weights().resize(out_sub_dim_lin, in_sub_dim);
		in_weights().init(init_strategy);

		rec_weights().resize(out_sub_dim_lin, out_sub_dim_lin);
		rec_weights().init(init_strategy);

		_biases.resize(out_sub_dim_lin);
		_biases.init(FillRandomUniform);
	}

	template<class D>
	RMLayer<D>::RMLayer(const int rec_depth, const Index3d& out_sub_dim, const typename D::matrix_t& in_w,
		const typename D::matrix_t& r_w, const typename D::vector_t& b, ActivationFunctionId func_id) :
		_rec_depth(rec_depth), _out_sub_dim(out_sub_dim)
	{
		if (in_w.row_dim() != b.dim() || r_w.row_dim() != b.dim() || r_w.row_dim() != r_w.col_dim())
			throw std::exception("Incompatible dimensions of weights and biases.");

		if (_out_sub_dim.coord_prod() != b.dim())
			throw std::exception("Invalid dimension of an output item.");

		set_func_id(func_id);
		in_weights() = in_w;
		rec_weights() = r_w;
		_biases = b;
	}

	template<class D>
	MLayerGradient<D> RMLayer<D>::allocate_gradient_container() const
	{
		MLayerGradient<D> result(1);

		auto& w = result[0].Weights_grad;
		w.resize(_weights.size());

		for (auto w_id = 0ull; w_id < w.size(); ++w_id)
			w[w_id].resize(_weights[w_id].size_3d());

		result[0].Biases_grad.resize(_biases.size_3d());

		return result;
	}

	template <class D>
	void RMLayer<D>::act(const IMLayerExchangeData<typename D::tensor_t>& input,
		IMLayerExchangeData<typename D::tensor_t>& output, IMLayerTraceData<D>* const trace_data) const
	{
		if (input.size() != _rec_depth || (trace_data && trace_data->size() != _rec_depth))
			throw std::exception("Invalid input data");

		output.resize(_rec_depth);

		for (auto iter_id = 0; iter_id < _rec_depth; iter_id++)
		{
			output[iter_id].resize(_out_sub_dim);

			in_weights().mul_add(input[iter_id], _biases, output[iter_id]);

			if (iter_id > 0)
				rec_weights().mul_add(output[iter_id - 1], output[iter_id], output[iter_id]);

			apply_activation_function(output[iter_id], trace_data ? &trace_data->item(iter_id).Derivatives : nullptr);
		}
	}

	template <class D>
	void RMLayer<D>::backpropagate(const IMLayerExchangeData<typename D::tensor_t>& out_grad,
		const IMLayerExchangeData<typename D::tensor_t>& output, const IMLayerExchangeData<LayerData<D>>& processing_data,
		IMLayerExchangeData<typename D::tensor_t>& out_input_grad, MLayerGradient<D>& out_layer_grad,
		const bool evaluate_input_gradient) const
	{
		if (out_grad.size() != _rec_depth || output.size() != _rec_depth ||
			processing_data.size() != _rec_depth || out_layer_grad.size() != 1 ||
			(evaluate_input_gradient && out_input_grad.size() != _rec_depth))
			throw std::exception("Invalid input.");

		thread_local typename D::tensor_t aux{};
		aux.resize(_out_sub_dim);
		aux.fill_zero();

		thread_local typename D::tensor_t temp{};
		temp.resize(_out_sub_dim);

		auto& biases_grad = out_layer_grad[0].Biases_grad;
		auto& in_weights_grad = out_layer_grad[0].Weights_grad[IN_W];
		auto& rec_weights_grad = out_layer_grad[0].Weights_grad[REC_W];

		biases_grad.fill_zero();
		in_weights_grad.fill_zero();
		rec_weights_grad.fill_zero();

		for (auto step_id = _rec_depth - 1; step_id >= 0; step_id--)
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
}
