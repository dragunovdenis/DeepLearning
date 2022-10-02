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

#include "NLayer.h"
#include "../Math/ActivationFunction.h"
#include <exception>

namespace DeepLearning
{
	template <class D>
	void NLayer<D>::initialize(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id,
		const Real rand_low, const Real rand_high, const bool standard_init_for_weights)
	{
		_biases = typename D::vector_t(out_dim, rand_low, rand_high);

		if (standard_init_for_weights)
		{
			_weights = typename D::matrix_t(out_dim, in_dim, false);
			_weights.standard_random_fill(Real(1) / std::sqrt(in_dim));
		}
		else
			_weights = typename D::matrix_t(out_dim, in_dim, rand_low, rand_high);

		_func_id = func_id;
	}

	template<class D>
	NLayer<D>::NLayer(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id,
		const Real rand_low, const Real rand_high, const bool standard_init_for_weights, const Real keep_rate) : ALayer<D>(keep_rate)
	{
		initialize(in_dim, out_dim, func_id, rand_low, rand_high, standard_init_for_weights);
	}

	template <class D>
	NLayer<D>::NLayer(const std::string& str) : ALayer<D>(str)
	{
		auto str_norm = Utils::normalize_string(str);

		Index3d temp;
		if (!Utils::try_extract_vector(str_norm, temp))
			throw std::exception("Can't parse input dimensions of NLayer");

		const auto in_dim = temp.coord_prod();

		if (!Utils::try_extract_vector(str_norm, temp))
			throw std::exception("Can't parse output dimensions of NLayer");

		if (temp.x != 1ll || temp.y != 1ll || temp.z <= 0ll)
			throw std::exception("Invalid output dimensions of NLayer");

		const auto out_dim = temp.z;

		const auto func_id = parse_activation_type(Utils::extract_word(str_norm));
		if (func_id == ActivationFunctionId::UNKNOWN)
			throw std::exception("Failed to parse activation function type of NLayer");

		initialize(in_dim, out_dim, func_id, Real(-1), Real(1), true);
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
	void NLayer<D>::act(const typename D::tensor_t& input, typename D::tensor_t& output, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr) const
	{
		const auto function = ActivationWrapper<typename D::tensor_t>(ActivationFunctionId(_func_id));

		output.resize(out_size());
		_weights.mul_add(input, _biases, output);

		if (aux_learning_data_ptr)
		{
			aux_learning_data_ptr->Input = input;
			function().func_and_aux_in_place(output, aux_learning_data_ptr->Derivatives);
		} else
			function().func_in_place(output);
	}

	template <class D>
	typename D::tensor_t NLayer<D>::act(const typename D::tensor_t& input, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr) const
	{
		typename D::tensor_t result;
		act(input, result, aux_learning_data_ptr);
		return result;
	}

	template <class D>
	void NLayer<D>::backpropagate(const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
		typename D::tensor_t& input_grad, typename ALayer<D>::LayerGradient& layer_grad, const bool evaluate_input_gradient) const
	{
		if (deltas.size_3d() != Index3d{ 1, 1, static_cast<long long>(_biases.dim()) })
			throw std::exception("Invalid input");

		const auto function = ActivationWrapper<typename D::tensor_t>(ActivationFunctionId(_func_id));
		function().calc_input_gradient(deltas, aux_learning_data.Derivatives, layer_grad.Biases_grad);
		layer_grad.Weights_grad.resize(1);
		vector_col_times_vector_row(layer_grad.Biases_grad, aux_learning_data.Input, layer_grad.Weights_grad[0]);

		if (!evaluate_input_gradient) return;

		_weights.transpose_mul(layer_grad.Biases_grad, input_grad);
		input_grad.reshape(aux_learning_data.Input.size_3d()); //Reshape, because inside this layer, we work with a "flattened" data,
															   //whereas previous layer might expect data of certain shape
	}

	template <class D>
	std::tuple<typename D::tensor_t, typename ALayer<D>::LayerGradient> NLayer<D>::backpropagate(const typename D::tensor_t& deltas,
		const typename ALayer<D>::AuxLearningData& aux_learning_data, const bool evaluate_input_gradient) const
	{
		typename D::tensor_t input_grad;
		typename ALayer<D>::LayerGradient layer_grad;
		backpropagate(deltas, aux_learning_data, input_grad, layer_grad, evaluate_input_gradient);
		return std::make_tuple(std::move(input_grad), std::move(layer_grad));
	}

	template <class D>
	void NLayer<D>::update(const std::tuple<std::vector<typename D::tensor_t>, typename D::tensor_t>& weights_and_biases_increment, const Real& reg_factor)
	{
		const auto& weights_increment = std::get<0>(weights_and_biases_increment);

		if (weights_increment.size() != 1)
			throw std::exception("Invalid input");

		if (reg_factor != Real(0))
			_weights.scale_and_add(weights_increment[0], Real(1) + reg_factor);
		else
			_weights.add(weights_increment[0]);

		_biases.add(std::get<1>(weights_and_biases_increment));
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
		return DeepLearning::to_string(NLayer::ID()) + "; Input size: " + in_size().to_string() + "; Out size: " + out_size().to_string() + 
			"; Activation: " + DeepLearning::to_string(_func_id) +";" + ALayer<D>::to_string();
	}

	template <class D>
	bool NLayer<D>::equal_hyperparams(const ALayer<D>& layer) const
	{
		const auto other_nlayer_ptr = dynamic_cast<const NLayer*>(&layer);
		return other_nlayer_ptr != nullptr && ALayer<D>::equal_hyperparams(layer)
		&& in_size() == layer.in_size() && out_size() == layer.out_size() && _func_id == other_nlayer_ptr->_func_id;
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
	std::string NLayer<D>::to_script() const
	{
		return in_size().to_string() + out_size().to_string() + ";" +
			DeepLearning::to_string(_func_id) + ";" + ALayer<D>::to_script();
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

	template <>
	NLayer<CpuDC> NLayer<CpuDC>::to_host() const
	{
		return *this;
	}

	template <>
	NLayer<CpuDC> NLayer<GpuDC>::to_host() const
	{
		NLayer<CpuDC> result;

		result._biases = _biases.to_host();
		result._weights = _weights.to_host();
		result._func_id = _func_id;

		return result;
	}

	template<>
	NLayer<GpuDC> NLayer<GpuDC>::to_device() const
	{
		return *this;
	}

	template<>
	NLayer<GpuDC> NLayer<CpuDC>::to_device() const
	{
		NLayer<GpuDC> result;

		result._biases.assign(_biases);
		result._weights.assign(_weights);
		result._func_id = _func_id;

		return result;
	}

	template class NLayer<CpuDC>;
	template class NLayer<GpuDC>;
}