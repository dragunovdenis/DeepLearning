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
#include "../Math/DenseVector.h"
#include "../Math/DenseMatrix.h"
#include "../Math/ActivationFunction.h"
#include "CummulativeGradient.h"
#include <msgpack.hpp>

namespace DeepLearning
{
	/// <summary>
	/// Representation of a single neural layer
	/// </summary>
	class NeuralLayer
	{
	public:
		/// <summary>
		/// Auxiliary data to perform learning on a level of single neuron layer
		/// </summary>
		struct AuxLearningData
		{
			/// <summary>
			/// Container to store input of a neuron layer
			/// </summary>
			DenseVector Input{};

			/// <summary>
			/// Container to store derivatives of the activation functions
			/// </summary>
			DenseVector Derivatives{};
		};

		/// <summary>
		/// A data structure to hold output of the back-propagation procedure
		/// </summary>
		struct LayerGradient
		{
			/// <summary>
			/// Gradient with respect to the biases of the neural layer
			/// </summary>
			DenseVector Biases_grad{};

			/// <summary>
			/// Gradient with respect to the weights of the neural layer
			/// </summary>
			DenseMatrix Weights_grad{};
		};

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
		/// Activation function id, use "unsigned int" instead of the enum in order to make msgpack happy
		/// </summary>
		unsigned int _func_id = ActivationFunctionId::UNKNOWN;

	public:

		MSGPACK_DEFINE(_biases, _weights, _func_id);

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
			const Real rand_low = Real(-1), const Real rand_high = Real(1));

		/// <summary>
		/// Constructor from the given weights and biases
		/// </summary>
		NeuralLayer(const DenseMatrix& weights, const DenseVector& biases, ActivationFunctionId func_id = ActivationFunctionId::SIGMOID);

		/// <summary>
		/// Copy constructor
		/// </summary>
		NeuralLayer(const NeuralLayer& anotherLayer);

		/// <summary>
		/// Makes a forward pass for the given input and outputs the result for the entire network
		/// </summary>
		/// <param name="input">Input signal</param>
		/// <param name="aux_learning_data_ptr">Pointer to the auxiliary data structure that should be provided during the training (learning) process</param>
		/// <returns>Output signal</returns>
		DenseVector act(const DenseVector& input, AuxLearningData* const aux_learning_data_ptr = nullptr) const;

		/// <summary>
		/// Performs the back-propagation
		/// </summary>
		/// <param name="deltas">Derivatives of the cost function with respect to the output of the current neural layer</param>
		/// <param name="aux_learning_data">Auxiliary learning data that should be obtained from the corresponding
		/// "forward" pass (see method "act") </param>
		/// <param name="evaluate_input_gradient">Determines whether the gradient with respect to the
		/// input data (the first item of the output tuple) will be actually evaluated.
		/// The evaluation is redundant for the very first layer of the net</param>
		/// <returns>Derivatives of the cost function with respect to the output of the previous neural layer
		/// (or input of the current neural layer, which is the same)</returns>
		std::tuple<DenseVector, LayerGradient> backpropagate(const DenseVector& deltas, const AuxLearningData& aux_learning_data,
			const bool evaluate_input_gradient = true) const;

		/// <summary>
		/// Adds given increments to the weights and biases respectively
		/// </summary>
		/// <param name="weights_and_biases_increment">Increment for weights and biases</param>
		/// <param name="reg_factor">Regularization factor, that (if non-zero) 
		/// results in term "reg_factor*w_i" being added to each weight "w_i" </param>
		void update(const std::tuple<DenseMatrix, DenseVector>& weights_and_biases_increment, const Real& reg_factor);
	};

}
