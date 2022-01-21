#pragma once
#include "../Math/DenseVector.h"
#include "../Math/DenseMatrix.h"
#include "../Math/ActivationFunction.h"
#include "AuxLearaningData.h"
#include <msgpack.hpp>

namespace DeepLearning
{
	/// <summary>
	/// Representation of a single neural layer
	/// </summary>
	class NeuralLayer
	{
	private:
		/// <summary>
		/// Vector of bias coefficients of size _out_dim;
		/// </summary>
		DenseVector _biases{};

		/// <summary>
		/// Matrix of weights of size _out_dim x _in_dim  
		/// </summary>
		DenseMatrix _weights{};

		/// <summary>
		/// Helper data structure to use in the learning process
		/// </summary>
		mutable std::unique_ptr<AuxLearningData> _learning_data{};

		/// <summary>
		/// Activation function id
		/// </summary>
		ActivationFunctionId _func_id = ActivationFunctionId::UNKNOWN;

	public:
		/// <summary>
		/// Dimensionality of the layer's input
		/// </summary>
		std::size_t in_dim() const;

		/// <summary>
		/// Dimensionality of the layer's output
		/// </summary>
		std::size_t out_dim() const;

		/// <summary>
		/// Default constructor
		/// </summary>
		NeuralLayer() = default;

		/// <summary>
		/// Constructor with random weights and biases
		/// </summary>
		NeuralLayer(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id = ActivationFunctionId::SIGMOID,
			const bool enable_learnign = false, const Real rand_low = Real(-1), const Real rand_high = Real(1));

		/// <summary>
		/// Constructor from the given weights and biases
		/// </summary>
		NeuralLayer(const DenseMatrix& weights, const DenseVector& biases, ActivationFunctionId func_id = ActivationFunctionId::SIGMOID, 
			const bool enable_learnign = false);

		/// <summary>
		/// Copy constructor
		/// </summary>
		NeuralLayer(const NeuralLayer& anotherLayer);

		/// <summary>
		/// Makes a forward pass for the given input and outputs the result for the entire network
		/// </summary>
		/// <param name="input">Input signal</param>
		/// <returns>Output signal</returns>
		DenseVector act(const DenseVector& input) const;

		/// <summary>
		/// Performs the back-propagation
		/// </summary>
		/// <param name="deltas">Derivatives of the cost function with respect to the output of the current neural layer</param>
		/// <returns>Derivatives of the cost function with respect to the output of the previous neural layer
		/// (or input of the current neural layer, which is the same)</returns>
		DenseVector backpropagate(const DenseVector& deltas, CummulativeLayerGradient& cumulative_gradient) const;

		/// <summary>
		/// Enables/disables learning mode for the neuron layer
		/// Enabling learning for the multiple times at a row acts as a "reset learning" action
		/// </summary>
		void enable_learning_mode(const bool learning);

		/// <summary>
		/// Adds given increments to the weights and biases respectively
		/// </summary>
		void update(const std::tuple<DenseMatrix, DenseVector>& weights_and_biases_increment);
	};

}
