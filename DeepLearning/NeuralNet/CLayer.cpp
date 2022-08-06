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

#include "CLayer.h"
#include <algorithm>
#include <numeric>
#include "../Diagnostics/Logging.h"
#include "../Math/ConvolutionUtils.h"

namespace DeepLearning
{
	template <class D>
	void CLayer<D>::initialize(const Index3d& in_size, const Index2d& filter_window_size,
		const std::size_t& filters_count, const ActivationFunctionId func_id, const Index3d& paddings, const Index3d& strides)
	{
		_in_size = in_size;
		_weight_tensor_size = { in_size.x, filter_window_size.x, filter_window_size.y };
		_paddings = paddings;
		_strides = strides;
		_func_id = func_id;

		const auto out_channel_size = ConvolutionUtils::calc_conv_res_size(_in_size, _weight_tensor_size, _paddings, _strides);
		if (out_channel_size.x != 1)
			throw std::exception("Unexpected channel size");

		_biases = typename D::tensor_t(filters_count, out_channel_size.y, out_channel_size.z, Real(-1), Real(1));
		_filters = std::vector(filters_count, typename D::tensor_t(_weight_tensor_size, false));

		std::for_each(_filters.begin(), _filters.end(), [](auto& filter) { filter.standard_random_fill(); });
	}

	template <class D>
	CLayer<D>::CLayer(const Index3d& in_size, const Index2d& filter_window_size,
		const std::size_t& filters_count, const ActivationFunctionId func_id, const Index3d& paddings, const Index3d& strides)
	{
		initialize(in_size, filter_window_size,	filters_count, func_id, paddings, strides);
	}

	template <class D>
	CLayer<D>::CLayer(const std::string& str)
	{
		auto str_norm = Utils::normalize_string(str);

		Index3d temp_3d;
		if (!Utils::try_extract_vector(str_norm, temp_3d))
			throw std::exception("Can't parse input dimensions of CLayer");

		const auto in_size = temp_3d;

		Index2d temp_2d;
		if (!Utils::try_extract_vector(str_norm, temp_2d))
			throw std::exception("Can't parse filter window size");

		const auto filter_window_size = temp_2d;

		const auto scalars = Utils::parse_scalars<long long>(Utils::extract_word(str_norm));

		if (scalars.size() != 1 || scalars[0] <= 0ll)
			throw std::exception("Can't parse number of filters");

		const auto filters_count = scalars[0];

		const auto func_id = parse_activation_type(Utils::extract_word(str_norm));

		if (func_id == ActivationFunctionId::UNKNOWN)
			throw std::exception("Failed to parse activation function type of CLayer");

		const auto paddings = Utils::try_extract_vector(str_norm, temp_3d) ? temp_3d : Index3d{ 0, 0, 0 };
		const auto strides  = Utils::try_extract_vector(str_norm, temp_3d) ? temp_3d : Index3d{ 1, 1, 1 };

		initialize(in_size, filter_window_size, filters_count, func_id, paddings, strides);
	}

	template <class D>
	void CLayer<D>::act(const typename D::tensor_t& input, typename D::tensor_t& output, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr) const
	{
		if (input.size_3d() != in_size())
			throw std::exception("Unexpected size of the input tensor");

		const auto function = ActivationWrapper<typename D::tensor_t>(ActivationFunctionId(_func_id));

		output.resize(out_size());
		input.convolve(output, _filters, _paddings, _strides);

		output += _biases;

		if (aux_learning_data_ptr)
		{
			aux_learning_data_ptr->Input = input;
			function().func_and_aux_in_place(output, aux_learning_data_ptr->Derivatives);
		} else
			function().func_in_place(output);
	}

	template <class D>
	typename D::tensor_t CLayer<D>::act(const typename D::tensor_t& input, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr) const
	{
		typename D::tensor_t result;
		act(input, result, aux_learning_data_ptr);
		return std::move(result);
	}

	template <class D>
	void CLayer<D>::backpropagate(const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
		typename D::tensor_t& input_grad, typename ALayer<D>::LayerGradient& layer_grad, const bool evaluate_input_gradient) const
	{
		if (deltas.size_3d() != aux_learning_data.Derivatives.size_3d())
			throw std::exception("Unexpected size of the input tensor of derivatives");

		const auto function = ActivationWrapper<typename D::tensor_t>(ActivationFunctionId(_func_id));
		function().calc_input_gradient(deltas, aux_learning_data.Derivatives, layer_grad.Biases_grad);

		if (layer_grad.Weights_grad.size() != _filters.size())
			layer_grad.Weights_grad.resize(_filters.size());

		auto& filters_grad = layer_grad.Weights_grad;

		if (evaluate_input_gradient)
		{
			input_grad.resize(in_size());
			input_grad.fill(Real(0));
		}
		const auto& input_tensor = aux_learning_data.Input;
		const auto& biases_grad = layer_grad.Biases_grad;

		if (evaluate_input_gradient)
			for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
			{
				input_tensor.template convolution_gradient<true>(
					static_cast<const typename D::tensor_t&>(biases_grad).get_layer_handle(filter_id), input_grad, filters_grad[filter_id],
					_filters[filter_id], _paddings, _strides);
			}
		else
			for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
			{
				input_tensor.template convolution_gradient<false>(
					static_cast<const typename D::tensor_t&>(biases_grad).get_layer_handle(filter_id), input_grad, filters_grad[filter_id],
					_filters[filter_id], _paddings, _strides);
			}
	}

	template <class D>
	std::tuple<typename D::tensor_t, typename ALayer<D>::LayerGradient> CLayer<D>::backpropagate(const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
		const bool evaluate_input_gradient) const
	{
		typename D::tensor_t input_grad;
		typename ALayer<D>::LayerGradient layer_grad;
		backpropagate(deltas, aux_learning_data, input_grad, layer_grad, evaluate_input_gradient);
		return std::make_tuple(std::move(input_grad), std::move(layer_grad));
	}

	template <class D>
	void CLayer<D>::update(const std::tuple<std::vector<typename D::tensor_t>, typename D::tensor_t>& weights_and_biases_increment, const Real& reg_factor)
	{
		const auto& weights_increment = std::get<0>(weights_and_biases_increment);

		if (reg_factor != Real(0))
			for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
				_filters[filter_id].scale_and_add(weights_increment[filter_id], Real(1) + reg_factor);
		else
			for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
				_filters[filter_id].add(weights_increment[filter_id]);

		_biases.add(std::get<1>(weights_and_biases_increment));
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
		return DeepLearning::to_string(CLayer::ID()) + "; Input size: " + in_size().to_string() +
			"; Out size: " + out_size().to_string() + "; Filter size: " + weight_tensor_size().to_string() +
			"; Activation: " + DeepLearning::to_string(_func_id);
	}

	template <class D>
	bool CLayer<D>::equal_hyperparams(const ALayer<D>& layer) const
	{
		const auto other_clayer_ptr = dynamic_cast<const CLayer*>(&layer);
		return other_clayer_ptr != nullptr && _in_size == layer.in_size() &&
			_weight_tensor_size == layer.weight_tensor_size() &&
			_paddings == other_clayer_ptr->_paddings &&
			_strides == other_clayer_ptr->_strides &&
			_func_id == other_clayer_ptr->_func_id;
	}

	template <class D>
	std::string CLayer<D>::to_script() const
	{
		return in_size().to_string() + weight_tensor_size().yz().to_string()
			+ ";" + Utils::to_string(out_size().x) + ";"
			+ DeepLearning::to_string(_func_id) + ";"+ _paddings.to_string() + _strides.to_string();
	}

	template <class D>
	LayerTypeId CLayer<D>::get_type_id() const
	{
		return ID();
	}

	template <class D>
	Real CLayer<D>::squared_weights_sum() const
	{
		return std::accumulate(_filters.begin(), _filters.end(), Real(0), [](const auto& sum, const auto& filter) { return sum + filter.sum_of_squares(); });
	}

	/// <summary>
	/// Copies all the field from the source to destination instance accept those that might
	/// require "host-to-device" or "device-to-host" copy operations
	/// </summary>
	template <class D>
	template <class D1, class D2>
	void CLayer<D>::partial_copy(const CLayer<D1>& src, CLayer<D2>& dest)
	{
		dest._in_size = src._in_size;
		dest._weight_tensor_size = src._weight_tensor_size;
		dest._paddings = src._paddings;
		dest._strides = src._strides;
		dest._func_id = src._func_id;
	}

	template <>
	CLayer<CpuDC> CLayer<CpuDC>::to_host() const
	{
		return *this;
	}

	template <>
	CLayer<CpuDC> CLayer<GpuDC>::to_host() const
	{
		CLayer<CpuDC> result;
		partial_copy(*this, result);
		result._biases = _biases.to_host();
		std::transform(_filters.begin(), _filters.end(), std::back_inserter(result._filters), [](const auto& x) { return x.to_host(); });

		return result;
	}

	template <>
	CLayer<GpuDC> CLayer<GpuDC>::to_device() const
	{
		return *this;
	}

	template <>
	CLayer<GpuDC> CLayer<CpuDC>::to_device() const
	{
		CLayer<GpuDC> result;
		partial_copy(*this, result);
		result._biases.assign(_biases);
		std::transform(_filters.begin(), _filters.end(), std::back_inserter(result._filters),
			[](const auto& x) 
			{
				GpuDC::tensor_t result;
				result.assign(x);
				return result; 
			});

		return result;
	}

	template class CLayer<CpuDC>;
	template class CLayer<GpuDC>;
}