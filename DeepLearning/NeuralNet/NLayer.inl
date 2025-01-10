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
#include "../Utilities.h"
#include <nlohmann/json.hpp>

namespace DeepLearning
{
	template <class D>
	void NLayer<D>::initialize(const std::size_t in_dim, const std::size_t out_dim,
		const Real rand_low, const Real rand_high, const bool standard_init_for_weights)
	{
		auto ran_gen_ptr = &ALayer<D>::ran_gen();
		_biases = typename D::vector_t(out_dim, rand_low, rand_high, ran_gen_ptr);

		if (standard_init_for_weights)
		{
			_weights = typename D::matrix_t(out_dim, in_dim, false);
			_weights.standard_random_fill(static_cast<Real>(1) / static_cast<Real>(std::sqrt(in_dim)), ran_gen_ptr);
		}
		else
			_weights = typename D::matrix_t(out_dim, in_dim, rand_low, rand_high, ran_gen_ptr);
	}

	template<class D>
	NLayer<D>::NLayer(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id,
		const Real rand_low, const Real rand_high, const bool standard_init_for_weights, const Real keep_rate) : ALayer<D>(keep_rate, func_id)
	{
		initialize(in_dim, out_dim, rand_low, rand_high, standard_init_for_weights);
	}

	template <class D>
	NLayer<D>::NLayer(const std::string& str, const Index3d& default_in_size) : ALayer<D>(str)
	{
		const auto json = nlohmann::json::parse(str);

		const auto in_size = json.contains(json_in_size_id()) ?
			Utils::extract_vector<Index3d>(json[json_in_size_id()].template get<std::string>()).coord_prod() :
			default_in_size.coord_prod();

		const auto out_size = json.contains(json_out_size_id()) ?
			Utils::extract_vector<Index3d>(json[json_out_size_id()].template get<std::string>()).coord_prod() :
			throw std::exception("Can't parse output dimensions of NLayer");

		initialize(in_size, out_size, static_cast<Real>(-1), static_cast<Real>(1), true);
	}

	template <class D>
	template <class D1>
	NLayer<D>::NLayer(const NLayer<D1>& source) : NLayer(source.to_script())
	{
		_biases = D::from_host(D1::to_host(source.biases()));
		_weights = D::from_host(D1::to_host(source.weights()));
	}

	template <class D>
	std::string NLayer<D>::to_script() const
	{
		nlohmann::json json = nlohmann::json::parse(ALayer<D>::to_script());;
		json[json_in_size_id()] = in_size().to_string();
		json[json_out_size_id()] = out_size().to_string();

		return json.dump();
	}

	template <class D>
	const typename D::vector_t& NLayer<D>::biases() const
	{
		return _biases;
	}

	template <class D>
	const typename D::matrix_t& NLayer<D>::weights() const
	{
		return _weights;
	}

	template <class D>
	void NLayer<D>::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		try
		{
			auto msg_pack_version = 0;
			msgpack::type::make_define_array(msg_pack_version, MSGPACK_BASE(ALayer<D>), _biases, _weights).msgpack_unpack(msgpack_o);
		} catch (...)
		{
			// to preserve backward compatibility
			Real keep_rate = - 1;
			auto func_id = ActivationFunctionId::UNKNOWN;
			msgpack::type::make_define_array(keep_rate, _biases, _weights, func_id).msgpack_unpack(msgpack_o);
			this->set_keep_rate(keep_rate);
			this->set_func_id(func_id);
		}
	}

	template <class D>
	Index3d NLayer<D>::in_size() const
	{
		return { 1ll, 1ll, static_cast<long long>(_weights.col_dim()) };
	}

	template <class D>
	Index3d NLayer<D>::out_size() const
	{
		return { 1ll, 1ll, static_cast<long long>(_weights.row_dim()) };
	}

	template <class D>
	Index3d NLayer<D>::weight_tensor_size() const
	{
		return { 1ll, static_cast<long long>(_weights.row_dim()), static_cast<long long>(_weights.col_dim()) };
	}

	template <class D>
	void NLayer<D>::act(const typename D::tensor_t& input, typename D::tensor_t& output, LayerTraceData<D>* const trace_data) const
	{
		output.resize(out_size());
		_weights.mul_add(input, _biases, output);

		if (trace_data)
		{
			this->get_func().func_and_aux_in_place(output, trace_data->Derivatives);
		} else
			this->get_func().func_in_place(output);
	}

	template <class D>
	void NLayer<D>::backpropagate(const typename D::tensor_t& deltas, const LayerData<D>& processing_data,
		typename D::tensor_t& input_grad, LayerGradient<D>& layer_grad, const bool evaluate_input_gradient,
		const Real gradient_scale_factor) const
	{
		if (deltas.size_3d() != Index3d{ 1, 1, static_cast<long long>(_biases.dim()) })
			throw std::exception("Invalid input");

		const auto nontrivial_scaling = gradient_scale_factor != static_cast<Real>(0);
		thread_local typename D::tensor_t bias_shared;
		auto& pure_bias_grad = nontrivial_scaling ? bias_shared.get_resized(out_size()) : layer_grad.data[0];

		this->get_func().calc_in_grad(deltas, processing_data.Trace.Derivatives, pure_bias_grad);

		if (nontrivial_scaling)
		{
			layer_grad.data[0].scale_and_add(pure_bias_grad, gradient_scale_factor);
			scale_and_add_vector_col_times_vector_row(pure_bias_grad, processing_data.Input,
				layer_grad.data[1], gradient_scale_factor);
		} else
			vector_col_times_vector_row(pure_bias_grad, processing_data.Input, layer_grad.data[1]);

		if (!evaluate_input_gradient) return;

		_weights.transpose_mul(pure_bias_grad, input_grad);
		input_grad.reshape(processing_data.Input.size_3d()); //Reshape, because inside this layer, we work with a "flattened" data,
															   //whereas previous layer might expect data of certain shape
	}

	template <class D>
	void NLayer<D>::allocate(LayerGradient<D>& gradient_container, bool fill_zeros) const
	{
		gradient_container.data.resize(2);
		gradient_container.data.shrink_to_fit();
		gradient_container.data[0].resize(_biases.size_3d());
		gradient_container.data[1].resize(_weights.size_3d());

		if (fill_zeros)
			gradient_container.fill_zero();
	}

	template <class D>
	void NLayer<D>::update(const LayerGradient<D>& gradient, const Real& l_rate, const Real& reg_factor)
	{
		const auto& data = gradient.data;

		if (data.size() != 2)
			throw std::exception("Invalid input");

		if (reg_factor != Real(0))
			_weights.scale_and_add_scaled(Real(1) + reg_factor, data[1], l_rate);
		else
			_weights.add_scaled(data[1], l_rate);

		_biases.add_scaled(data[0], l_rate);
	}

	template <class D>
	void NLayer<D>::log(const std::filesystem::path& directory) const
	{
		if (!std::filesystem::is_directory(directory))
			throw std::exception("Directory does not exist");

		_weights.log(directory / "weights.txt");
		_biases.log(directory / "biases.txt");
	}

	template <class D>
	std::string NLayer<D>::to_string() const
	{
		return DeepLearning::to_string(NLayer::ID()) + "; " + to_script();
	}

	template <class D>
	bool NLayer<D>::equal_hyperparams(const ALayer<D>& layer) const
	{
		const auto other_nlayer_ptr = dynamic_cast<const NLayer*>(&layer);
		return other_nlayer_ptr != nullptr && ALayer<D>::equal_hyperparams(layer)
		&& in_size() == layer.in_size() && out_size() == layer.out_size();
	}

	template <class D>
	bool NLayer<D>::equal(const ALayer<D>& layer) const
	{
		if (!equal_hyperparams(layer))
			return false;

		//no need to check if the casted value is null because the check is done in the "hyperparams" function
		const auto other_nlayer_ptr = dynamic_cast<const NLayer*>(&layer); 
		return other_nlayer_ptr->_weights == _weights && other_nlayer_ptr->_biases == _biases;
	}

	template <class D>
	LayerTypeId NLayer<D>::get_type_id() const
	{
		return ID();
	}

	template <class D>
	Real NLayer<D>::squared_weights_sum() const
	{
		return _weights.sum_of_squares();
	}

	template <class D>
	template <class D1>
	NLayer<D1> NLayer<D>::convert() const
	{
		if (std::is_same_v<D1, D>)
			return *this;

		return NLayer<D1>(*this);
	}

	template <class D>
	void NLayer<D>::reset()
	{
		_weights.fill_zero();
		_biases.fill_zero();
	}
}