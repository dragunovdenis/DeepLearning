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
#include "../Diagnostics/Logging.h"

namespace DeepLearning
{
	CLayer::CLayer(const Index3d& in_size, const Index2d& filter_window_size,
		const std::size_t& filters_count, const ActivationFunctionId func_id, const Index3d& paddings, const Index3d& strides) :_in_size(in_size),
		_weight_tensor_size(in_size.x, filter_window_size.x, filter_window_size.y), _paddings(paddings), _strides(strides), _func_id(func_id)
	{
		const auto out_channel_size = Tensor::calc_conv_res_size(_in_size, _weight_tensor_size, _paddings, _strides);
		if (out_channel_size.x != 1)
			throw std::exception("Unexpected channel size");

		_biases = Tensor(filters_count, out_channel_size.y, out_channel_size.z, Real(-1), Real(1));
		_filters = std::vector(filters_count, Tensor(_weight_tensor_size, false));

		std::for_each(_filters.begin(), _filters.end(), [](auto& filter) { filter.standard_random_fill(); });
	}

	Tensor CLayer::act(const Tensor& input, AuxLearningData* const aux_learning_data_ptr) const
	{
		if (input.size_3d() != in_size())
			throw std::exception("Unexpected size of the input tensor");

		const auto function = ActivationFuncion(ActivationFunctionId(_func_id));

		auto temp = Tensor(out_size(), false);

		for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
			input.convolve(temp.get_layer_handle(filter_id), _filters[filter_id], _paddings, _strides);

		temp += _biases;

		if (aux_learning_data_ptr)
		{
			aux_learning_data_ptr->Input = input;
			auto [result, deriv] = function.func_and_deriv(temp);
			aux_learning_data_ptr->Derivatives = std::move(deriv);
			return std::move(result);
		}

		return std::move(function(temp));
	}

	std::tuple<Tensor, CLayer::LayerGradient> CLayer::backpropagate(const Tensor& deltas, const AuxLearningData& aux_learning_data,
		const bool evaluate_input_gradient) const
	{
		if (deltas.size_3d() != aux_learning_data.Derivatives.size_3d())
			throw std::exception("Unexpected size of the input tensor of derivatives");

		auto biases_grad = deltas.hadamard_prod(aux_learning_data.Derivatives);

		auto filters_grad = std::vector<Tensor>(_filters.size());
		Tensor input_grad(in_size(), true);
		const auto& input_tensor = aux_learning_data.Input;

		for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
		{
			auto [filter_grad, input_grad_partial] = input_tensor.convolution_gradient(
				static_cast<const Tensor&>(biases_grad).get_layer_handle(filter_id), _filters[filter_id], _paddings, _strides);

			if (evaluate_input_gradient)
				input_grad += input_grad_partial;

			filters_grad[filter_id] = std::move(filter_grad);
		}

		return std::make_tuple<Tensor, CLayer::LayerGradient>(std::move(input_grad), { std::move(biases_grad), std::move(filters_grad) });
	}

	void CLayer::update(const std::tuple<std::vector<Tensor>, Tensor>& weights_and_biases_increment, const Real& reg_factor)
	{
		const auto& weights_increment = std::get<0>(weights_and_biases_increment);

		if (reg_factor != Real(0))
			for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
				_filters[filter_id].add_scaled(weights_increment[filter_id], _filters[filter_id], reg_factor);
		else
			for (auto filter_id = 0ull; filter_id < _filters.size(); filter_id++)
				_filters[filter_id].add(weights_increment[filter_id]);

		_biases.add(std::get<1>(weights_and_biases_increment));
	}

	Index3d CLayer::in_size() const
	{
		return _in_size;
	}

	Index3d CLayer::out_size() const
	{
		return _biases.size_3d();
	}

	Index3d CLayer::weight_tensor_size() const
	{
		return _weight_tensor_size;
	}

	void CLayer::log(const std::filesystem::path& directory) const
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
}