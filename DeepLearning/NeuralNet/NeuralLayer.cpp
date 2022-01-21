#include "NeuralLayer.h"
#include "../Math/ActivationFunction.h"
#include <exception>

namespace DeepLearning
{
	NeuralLayer::NeuralLayer(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id, const bool enable_learnign,
		const Real rand_low, const Real rand_high)
	{
		_biases  = DenseVector(out_dim, rand_low, rand_high);
		_weights = DenseMatrix(out_dim, in_dim, rand_low, rand_high);
		_func_id = func_id;

		enable_learning_mode(enable_learnign);
	}

	NeuralLayer::NeuralLayer(const DenseMatrix& weights, const DenseVector& biases, ActivationFunctionId func_id, const bool enable_learnign)
	{
		if (weights.row_dim() != biases.dim())
			throw std::exception("incompatible dimensions of the weight and biases containers");

		_biases = biases;
		_weights = weights;
		_func_id = func_id;

		enable_learning_mode(enable_learnign);
	}

	NeuralLayer::NeuralLayer(const NeuralLayer& anotherLayer)
	{
		_biases = anotherLayer._biases;
		_weights = anotherLayer._weights;
		_func_id = anotherLayer._func_id;

		enable_learning_mode(_learning_data != nullptr);
	}

	std::size_t NeuralLayer::in_dim() const
	{
		return _weights.col_dim();
	}

	std::size_t NeuralLayer::out_dim() const
	{
		return _weights.row_dim();
	}

	DenseVector NeuralLayer::act(const DenseVector& input) const
	{
		const auto function = ActivationFuncion(_func_id);

		const auto z = _weights * input + _biases;

		if (_learning_data)
		{
			_learning_data->Input = input;
			const auto [result, deriv] = function.func_and_deriv(z);
			_learning_data->Derivatives = deriv;
			return result;
		}

		return function(z);
	}

	DenseVector NeuralLayer::backpropagate(const DenseVector& deltas, CummulativeLayerGradient& cumulative_gradient) const
	{
		if (!_learning_data)
			throw std::exception("Learning data is invalid");

		const auto biases_ghrad = deltas.hadamard_prod(_learning_data->Derivatives);
		const auto weights_grad = vector_col_times_vector_row(biases_ghrad, _learning_data->Input);

		cumulative_gradient.Add(weights_grad, biases_ghrad);

		return biases_ghrad * _weights;
	}

	void NeuralLayer::enable_learning_mode(const bool learning)
	{
		if (learning)
			_learning_data = std::make_unique<AuxLearningData>();
		else
			_learning_data.reset();
	}

	void NeuralLayer::update(const std::tuple<DenseMatrix, DenseVector>& weights_and_biases_increment)
	{
		_weights += std::get<0>(weights_and_biases_increment);
		_biases += std::get<1>(weights_and_biases_increment);
	}
}