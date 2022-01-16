#pragma once
#include "../Math/DenseVector.h"
#include "../Math/DenseMatrix.h"
#include "../Math/ActivationFunction.h"
#include <msgpack.hpp>

namespace DeepLearning
{
	class AuxLearningData;
	class ActivationFuncion;

	/// <summary>
	/// Representation of a single neural layer
	/// </summary>
	class NeuralLayer
	{
	private:
		/// <summary>
		/// Dimensionality of the layer's input
		/// </summary>
		std::size_t _in_dim;

		/// <summary>
		/// Dimensionality of the layer's output
		/// </summary>
		std::size_t _out_dim;

		/// <summary>
		/// Vector of bias coefficients of size _out_dim;
		/// </summary>
		DenseVector _biases{};

		/// <summary>
		/// Matrix of weights of size _out_dim x _in_dim  
		/// </summary>
		DenseMatrix _weights{};

		/// <summary>
		/// Pointer to the previous layer in the network (used for backward propagation)
		/// </summary>
		NeuralLayer* _prev_layer{};

		/// <summary>
		/// Pointer to the next layer in the network (used for forward pass)
		/// </summary>
		NeuralLayer* _next_layer{};

		/// <summary>
		/// Helper data structure to use in the learning process
		/// </summary>
		std::unique_ptr<AuxLearningData> _learning_data{};

		/// <summary>
		/// Activation function id
		/// </summary>
		ActivationFunctionId _func_id = ActivationFunctionId::UNKNOWN;
		
		/// <summary>
		/// The activation function
		/// </summary>
		std::unique_ptr<ActivationFuncion> _function{};

		/// <summary>
		/// Instantiates activation function according to the current function id
		/// </summary>
		std::unique_ptr<ActivationFuncion> instantiate_activation_function() const;

	public:
		/// <summary>
		/// Default constructor
		/// </summary>
		NeuralLayer() = default;

		/// <summary>
		/// Constructor with random weights and biases
		/// </summary>
		NeuralLayer(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id = ActivationFunctionId::SIGMOID);

		/// <summary>
		/// Constructor from the given weights and biases
		/// </summary>
		NeuralLayer(const DenseMatrix& weights, const DenseVector& biases, ActivationFunctionId func_id = ActivationFunctionId::SIGMOID);

		/// <summary>
		/// Makes a forward pass for the given input and outputs the result for the entire network
		/// </summary>
		/// <param name="input">Input signal</param>
		/// <returns>Output signal</returns>
		DenseVector act(const DenseVector& input);

		/// <summary>
		/// Performs the back-propagation
		/// </summary>
		/// <param name="deltas">Derivatives of the cost function with respect to the output of the current neural layer</param>
		/// <returns>Derivatives of the cost function with respect to the output of the previous neural layer
		/// (or input of the current neural layer, which is the same)</returns>
		DenseVector backpropagate(const DenseVector& deltas);

		/// <summary>
		/// Enables/disables learning mode for the neuron layer
		/// </summary>
		void enable_learning_mode(const bool learning);
	};

}
