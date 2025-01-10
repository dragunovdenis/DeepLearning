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

#include <algorithm>
#include <numeric>
#include "../Diagnostics/Logging.h"
#include "../Math/ConvolutionUtils.h"
#include "../Utilities.h"
#include <nlohmann/json.hpp>

namespace DeepLearning
{
	template <class D>
	void CLayer<D>::initialize(const Index3d& in_size, const Index2d& filter_window_size,
		const std::size_t& filters_count, const Index3d& paddings, const Index3d& strides)
	{
		_in_size = in_size;
		_weight_tensor_size = { in_size.x, filter_window_size.x, filter_window_size.y };
		_paddings = paddings;
		_strides = strides;

		const auto out_channel_size = ConvolutionUtils::calc_conv_res_size(_in_size, _weight_tensor_size, _paddings, _strides);
		if (out_channel_size.x != 1)
			throw std::exception("Unexpected channel size");

		auto ran_gen_ptr = &ALayer<D>::ran_gen();
		_biases = typename D::tensor_t(filters_count, out_channel_size.y, out_channel_size.z,
			Real(-1), Real(1), ran_gen_ptr);
		_filters = std::vector(filters_count, typename D::tensor_t(_weight_tensor_size, false));

		std::for_each(_filters.begin(), _filters.end(), [ran_gen_ptr](auto& filter)
			{ filter.standard_random_fill(/*sigma*/ -1, ran_gen_ptr); });
	}

	template <class D>
	CLayer<D>::CLayer(const Index3d& in_size, const Index2d& filter_window_size,
		const std::size_t& filters_count, const ActivationFunctionId func_id,
		const Index3d& paddings, const Index3d& strides, const Real keep_rate) : ALayer<D>(keep_rate, func_id)
	{
		initialize(in_size, filter_window_size, filters_count, paddings, strides);
	}

	template <class D>
	CLayer<D>::CLayer(const std::string& str, const Index3d& default_in_size) : ALayer<D>(str)
	{
		const auto json = nlohmann::json::parse(str);

		const auto in_size = json.contains(json_in_size_id()) ?
			Utils::extract_vector<Index3d>(json[json_in_size_id()].template get<std::string>()) : default_in_size;

		const auto filter_window_size = json.contains(json_filter_window_size_id()) ?
			Utils::extract_vector<Index2d>(json[json_filter_window_size_id()].template get<std::string>()) :
			throw std::exception("Can't parse filter window size of CLayer");

		const auto filters_count = json.contains(json_filters_count_id()) ?
			json[json_filters_count_id()].template get<std::size_t>() :
			throw std::exception("Can't parse number of filters of CLayer");

		const auto paddings = json.contains(json_paddings_id()) ?
			Utils::extract_vector<Index3d>(json[json_paddings_id()].template get<std::string>()) : Index3d{0, 0, 0};

		const auto strides = json.contains(json_strides_id()) ?
			Utils::extract_vector<Index3d>(json[json_strides_id()].template get<std::string>()) : Index3d{0, 0, 0};

		initialize(in_size, filter_window_size, filters_count, paddings, strides);
	}

	template <class D>
	template <class D1>
	CLayer<D>::CLayer(const CLayer<D1>& source) : CLayer(source.to_script())
	{
		_biases = D::from_host(D1::to_host(source.biases()));
		const auto& source_filters = source.filters();
		std::ranges::transform(source_filters, _filters.begin(),
			[](const auto& x) { return D::from_host(D1::to_host(x)); });
	}

	template <class D>
	std::string CLayer<D>::to_script() const
	{
		nlohmann::json json = nlohmann::json::parse(ALayer<D>::to_script());

		json[json_in_size_id()] = in_size().to_string();
		json[json_filter_window_size_id()] = weight_tensor_size().yz().to_string();
		json[json_filters_count_id()] = out_size().x;
		json[json_paddings_id()] = _paddings.to_string();
		json[json_strides_id()] = _strides.to_string();

		return json.dump();
	}

	template <class D>
	void CLayer<D>::act(const typename D::tensor_t& input, typename D::tensor_t& output, LayerTraceData<D>* const trace_data) const
	{
		if (input.size_3d() != in_size())
			throw std::exception("Unexpected size of the input tensor");

		output.resize(out_size());
		input.convolve(output, _filters, _paddings, _strides);

		output += _biases;

		if (trace_data)
		{
			this->get_func().func_and_aux_in_place(output, trace_data->Derivatives);
		} else
			this->get_func().func_in_place(output);
	}

	template <class D>
	void CLayer<D>::backpropagate(const typename D::tensor_t& deltas, const LayerData<D>& processing_data,
		typename D::tensor_t& input_grad, LayerGradient<D>& layer_grad, const bool evaluate_input_gradient,
		const Real gradient_scale_factor) const
	{
		auto& derivatives = processing_data.Trace.Derivatives;
		if (deltas.size_3d() != derivatives.size_3d())
			throw std::exception("Unexpected size of the input tensor of derivatives");

		const auto nontrivial_scaling = gradient_scale_factor != static_cast<Real>(0);
		thread_local typename D::tensor_t bias_shared;
		auto& pure_bias_grad = nontrivial_scaling ? bias_shared.
			get_resized(layer_grad.data[0].size_3d()) : layer_grad.data[0];

		this->get_func().calc_in_grad(deltas, derivatives, pure_bias_grad);

		if (nontrivial_scaling)
			layer_grad.data[0].scale_and_add(pure_bias_grad, gradient_scale_factor);

		auto& filters_grad = layer_grad.data;

		if (evaluate_input_gradient)
		{
			input_grad.resize(in_size());
			input_grad.fill_zero();
		}

		const auto& input_tensor = processing_data.Input;

		for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
		{
			if (evaluate_input_gradient)
				input_tensor.template convolution_gradient<true>(
					static_cast<const typename D::tensor_t&>(pure_bias_grad).get_layer_handle(filter_id),
					input_grad, filters_grad[filter_id + 1], _filters[filter_id], _paddings, _strides, gradient_scale_factor);
			else
				input_tensor.template convolution_gradient<false>(
					static_cast<const typename D::tensor_t&>(pure_bias_grad).get_layer_handle(filter_id),
					input_grad, filters_grad[filter_id + 1], _filters[filter_id], _paddings, _strides, gradient_scale_factor);
		}
	}

	template <class D>
	void CLayer<D>::allocate(LayerGradient<D>& gradient_container, bool fill_zeros) const
	{
		gradient_container.data.resize(_filters.size() + 1);
		gradient_container.data.shrink_to_fit();
		gradient_container.data[0].resize(_biases.size_3d());
		auto& grad_data = gradient_container.data;

		for (auto filter_id = 1ull; filter_id < grad_data.size(); ++filter_id)
			grad_data[filter_id].resize(_filters[filter_id - 1].size_3d());

		if (fill_zeros)
			gradient_container.fill_zero();
	}

	template <class D>
	void CLayer<D>::update(const LayerGradient<D>& gradient, const Real& l_rate, const Real& reg_factor)
	{
		const auto& data = gradient.data;

		if (reg_factor != static_cast<Real>(0))
			for (auto filter_id = 1ull; filter_id < data.size(); filter_id++)
				_filters[filter_id - 1].scale_and_add_scaled(Real(1) + reg_factor, data[filter_id], l_rate);
		else
			for (auto filter_id = 1ull; filter_id < data.size(); filter_id++)
				_filters[filter_id - 1].add_scaled(data[filter_id], l_rate);

		_biases.add_scaled(gradient.data[0], l_rate);
	}

	template <class D>
	const typename D::tensor_t& CLayer<D>::biases() const
	{
		return _biases;
	}

	template <class D>
	const std::vector<typename D::tensor_t>& CLayer<D>::filters() const
	{
		return _filters;
	}

	template <class D>
	void CLayer<D>::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		try
		{
			auto msg_pack_version = 0;
			msgpack::type::make_define_array(msg_pack_version, MSGPACK_BASE(ALayer<D>),
				_in_size, _weight_tensor_size, _paddings, _strides, _biases, _filters).msgpack_unpack(msgpack_o);
		}
		catch (...)
		{
			// to preserve backward compatibility
			Real keep_rate = -1;
			auto func_id = ActivationFunctionId::UNKNOWN;
			msgpack::type::make_define_array(keep_rate, _in_size, _weight_tensor_size,
				_paddings, _strides, func_id, _biases, _filters).msgpack_unpack(msgpack_o);
			this->set_keep_rate(keep_rate);
			this->set_func_id(func_id);
		}
	}

	template <class D>
	Index3d CLayer<D>::in_size() const
	{
		return _in_size;
	}

	template <class D>
	Index3d CLayer<D>::out_size() const
	{
		return _biases.size_3d();
	}

	template <class D>
	Index3d CLayer<D>::weight_tensor_size() const
	{
		return _weight_tensor_size;
	}

	template <class D>
	void CLayer<D>::log(const std::filesystem::path& directory) const
	{
		if (!std::filesystem::is_directory(directory))
			throw std::exception("Directory does not exist");

		const auto biases_folder = directory / "biases";
		Logging::make_path(biases_folder);

		_biases.log(biases_folder, "channel");

		for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
		{
			const auto filter_folder = directory / (std::string("filter_") + std::to_string(filter_id));
			Logging::make_path(filter_folder);

			_filters[filter_id].log(filter_folder, "channel");
		}
	}

	template <class D>
	std::string CLayer<D>::to_string() const
	{
		return DeepLearning::to_string(ID()) + "; " + to_script();
	}

	template <class D>
	bool CLayer<D>::equal_hyperparams(const ALayer<D>& layer) const
	{
		const auto other_clayer_ptr = dynamic_cast<const CLayer*>(&layer);
		return other_clayer_ptr != nullptr &&
			ALayer<D>::equal_hyperparams(layer) && _in_size == layer.in_size() &&
			_weight_tensor_size == layer.weight_tensor_size() &&
			_paddings == other_clayer_ptr->_paddings &&
			_strides == other_clayer_ptr->_strides;
	}

	template <class D>
	bool CLayer<D>::equal(const ALayer<D>& layer) const
	{
		if (!equal_hyperparams(layer))
			return false;

		//no need to check if the casted value is null because the check is done in the "hyperparams" function
		const auto other_nlayer_ptr = dynamic_cast<const CLayer*>(&layer);
		return other_nlayer_ptr->_filters == _filters && other_nlayer_ptr->_biases == _biases;
	}

	template <class D>
	LayerTypeId CLayer<D>::get_type_id() const
	{
		return ID();
	}

	template <class D>
	Real CLayer<D>::squared_weights_sum() const
	{
		return std::accumulate(_filters.begin(), _filters.end(), static_cast<Real>(0),
			[](const auto& sum, const auto& filter) { return sum + filter.sum_of_squares(); });
	}

	template <class D>
	template <class D1>
	CLayer<D1> CLayer<D>::convert() const
	{
		if (std::is_same_v<D1, D>)
			return *this;

		return CLayer<D1>(*this);
	}

	template <class D>
	void CLayer<D>::reset()
	{
		std::ranges::for_each(_filters,
			[](typename D::tensor_t& filter)
			{ filter.fill_zero(); });

		_biases.fill_zero();
	}
}