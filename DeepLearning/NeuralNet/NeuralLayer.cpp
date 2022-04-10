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

#include "NeuralLayer.h"
#include "../Math/ActivationFunction.h"
#include <exception>

namespace DeepLearning
{
	NeuralLayer::NeuralLayer(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id,
		const Real rand_low, const Real rand_high)
	{
		_biases  = Vector(out_dim, rand_low, rand_high);
		_weights = Matrix(out_dim, in_dim, rand_low, rand_high);
		_func_id = static_cast<unsigned int>(func_id);
	}

	NeuralLayer::NeuralLayer(const Matrix& weights, const Vector& biases, ActivationFunctionId func_id)
	{
		if (weights.row_dim() != biases.dim())
			throw std::exception("incompatible dimensions of the weight and biases containers");

		_biases = biases;
		_weights = weights;
		_func_id = static_cast<unsigned int>(func_id);
	}

	NeuralLayer::NeuralLayer(const NeuralLayer& anotherLayer)
	{
		_biases = anotherLayer._biases;
		_weights = anotherLayer._weights;
		_func_id = anotherLayer._func_id;
	}

	std::size_t NeuralLayer::in_dim() const
	{
		return _weights.col_dim();
	}

	std::size_t NeuralLayer::out_dim() const
	{
		return _weights.row_dim();
	}

	Vector NeuralLayer::act(const Vector& input, AuxLearningData* const aux_learning_data_ptr) const
	{
		const auto function = ActivationFuncion(ActivationFunctionId(_func_id));

		const auto z = _weights.mul_add(input, _biases);

		if (aux_learning_data_ptr)
		{
			aux_learning_data_ptr->Input = input;
			const auto [result, deriv] = function.func_and_deriv(z);
			aux_learning_data_ptr->Derivatives = deriv;
			return result;
		}

		return function(z);
	}

	std::tuple<Vector, NeuralLayer::LayerGradient> NeuralLayer::backpropagate(const Vector& deltas,
		const AuxLearningData& aux_learning_data, const bool evaluate_input_gradient) const
	{
		const auto biases_ghrad = deltas.hadamard_prod(aux_learning_data.Derivatives);
		const auto weights_grad = vector_col_times_vector_row(biases_ghrad, aux_learning_data.Input);

		return std::make_tuple<Vector, NeuralLayer::LayerGradient>(
			evaluate_input_gradient ? biases_ghrad * _weights : Vector(0),
			{ biases_ghrad, weights_grad });
	}

	void NeuralLayer::update(const std::tuple<Matrix, Vector>& weights_and_biases_increment, const Real& reg_factor)
	{
		if (reg_factor != Real(0))
			_weights += (std::get<0>(weights_and_biases_increment) + _weights * reg_factor);
		else
			_weights += std::get<0>(weights_and_biases_increment);

		_biases += std::get<1>(weights_and_biases_increment);
	}
}