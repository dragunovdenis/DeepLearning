#include "NeuralLayer.h"
#include "AuxLearaningData.h"
#include "../Math/ActivationFunction.h"
#include <exception>

namespace DeepLearning
{
	NeuralLayer::NeuralLayer(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id)
		: _in_dim(in_dim), _out_dim(out_dim)
	{
		_biases  = DenseVector(_out_dim, -1, 1);
		_weights = DenseMatrix(_out_dim, _in_dim, -1, 1);
		_func_id = func_id;
	}

	DenseVector NeuralLayer::act(const DenseVector& input)
	{
		if (!_function)
			_function = instantiate_activation_function();

		const auto z = _weights * input + _biases;

		if (_learning_data)
		{
			_learning_data->Input = input;
			const auto [result, deriv] = _function->func_and_deriv(z);
			_learning_data->Derivatives = deriv;
			return result;
		}

		return (*_function)(z);
	}

	DenseVector NeuralLayer::backpropagate(const DenseVector& deltas)
	{
		if (!_learning_data)
			throw std::exception("Learning data is invalid");

		const auto biases_ghrad = deltas.hadamard_prod(_learning_data->Derivatives);
		const auto weights_grad = vector_col_times_vector_row(biases_ghrad, _learning_data->Input);
		_learning_data->add_gradient_contribution(weights_grad, biases_ghrad);

		return biases_ghrad * _weights;

	}

	void NeuralLayer::enable_learning_mode(const bool learning)
	{
		if (learning)
			_learning_data = std::make_unique<AuxLearningData>(_weights.row_dim(), _weights.col_dim());
		else
			_learning_data.reset();
	}

	std::unique_ptr<ActivationFuncion> NeuralLayer::instantiate_activation_function() const
	{
		switch (_func_id)
		{
			case ActivationFunctionId::UNKNOWN: throw std::exception("Invalid activation function ID.");
				break;
			case ActivationFunctionId::SIGMOID: return std::make_unique<Sigmoid>();
				break;
			default: throw std::exception("Unexpected activation function ID.");
				break;
		}
	}

}