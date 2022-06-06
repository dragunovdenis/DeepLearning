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
#include <vector>
#include "../Math/Tensor.h"
#include "CummulativeGradient.h"
#include <filesystem>
#include <string>
#include "LayerTypeId.h"

namespace DeepLearning
{
	/// <summary>
	/// An abstract neural net layer
	/// </summary>
	class ALayer
	{
	public:
		/// <summary>
		/// Auxiliary data to perform learning on a level of single convolution layer
		/// </summary>
		struct AuxLearningData
		{
			/// <summary>
			/// Container to store input of a convolution layer
			/// </summary>
			Tensor Input{};

			/// <summary>
			/// Container to store derivatives of the activation function
			/// </summary>
			Tensor Derivatives{};

			/// <summary>
			/// Place holder for index mappings
			/// </summary>
			std::vector<std::size_t> IndexMapping{};
		};

		/// <summary>
		/// A data structure to hold output of the back-propagation procedure
		/// </summary>
		struct LayerGradient
		{
			/// <summary>
			/// Gradient with respect to the biases of the convolution layer
			/// </summary>
			Tensor Biases_grad{};

			/// <summary>
			/// Gradient with respect to the weights of the convolution layer
			/// </summary>
			std::vector<Tensor> Weights_grad{};
		};

	public:

		/// <summary>
		/// Size of the layer's input
		/// </summary>
		virtual Index3d in_size() const = 0;

		/// <summary>
		/// Size of the layer's output
		/// </summary>
		virtual Index3d out_size() const = 0;

		/// <summary>
		/// Returns size of a single weights tensor
		/// </summary>
		virtual Index3d weight_tensor_size() const = 0;

		/// <summary>
		/// Makes a forward pass for the given input and outputs the result of the layer
		/// </summary>
		/// <param name="input">Input signal</param>
		/// <param name="aux_learning_data_ptr">Pointer to the auxiliary data structure that should be provided during the training (learning) process</param>
		/// <returns>Output signal</returns>
		virtual Tensor act(const Tensor& input, AuxLearningData* const aux_learning_data_ptr = nullptr) const = 0;

		/// <summary>
		/// Performs the back-propagation
		/// </summary>
		/// <param name="deltas">Derivatives of the cost function with respect to the output of the current layer</param>
		/// <param name="aux_learning_data">Auxiliary learning data that should be obtained from the corresponding
		/// "forward" pass (see method "act") </param>
		/// <param name="evaluate_input_gradient">Determines whether the gradient with respect to the
		/// input data (the first item of the output tuple) will be actually evaluated.
		/// The evaluation is redundant for the very first layer of the net</param>
		/// <returns>Derivatives of the cost function with respect to the output of the previous neural layer
		/// (or input of the current neural layer, which is the same) as well as the derivatives of the cost function
		/// with respect to the weight and biases of the current layer</returns>
		virtual std::tuple<Tensor, LayerGradient> backpropagate(const Tensor& deltas, const AuxLearningData& aux_learning_data,
			const bool evaluate_input_gradient = true) const = 0;

		/// <summary>
		/// Adds given increments to the weights and biases respectively
		/// </summary>
		/// <param name="weights_and_biases_increment">Increment for weights and biases</param>
		/// <param name="reg_factor">Regularization factor, that (if non-zero) 
		/// results in term "reg_factor*w_i" being added to each weight "w_i" </param>
		virtual void update(const std::tuple<std::vector<Tensor>, Tensor>& weights_and_biases_increment, const Real& reg_factor) = 0;

		/// <summary>
		/// Returns zero initialized instance of cumulative gradient suitable for the current instance of the layer
		/// </summary>
		virtual CummulativeGradient init_cumulative_gradient() const
		{
			return CummulativeGradient(weight_tensor_size(), out_size());
		}

		/// <summary>
		/// Logs layer to the given directory (for diagnostic purposes)
		/// </summary>
		/// <param name="directory">Directory to log to</param>
		virtual void log(const std::filesystem::path& directory) const = 0;

		/// <summary>
		/// Virtual destructor to ensure that the resources of descending classes are properly released
		/// </summary>
		virtual ~ALayer() {}

		/// <summary>
		/// Returns a human-readable description of the layer
		/// </summary>
		virtual std::string to_string() const = 0;

		/// <summary>
		/// Encodes hyper-parameters of the layer in a string-script which then can be used to instantiate 
		/// another instance of the layer with the same set of hyper-parameters (see the constructor taking string argument)
		/// </summary>
		virtual std::string to_script() const = 0;

		/// <summary>
		/// Returns identifier of the layer's type
		/// </summary>
		virtual LayerTypeId get_type_id() const = 0;

		/// <summary>
		/// Returns "true" if the current instance of the layer has the same set of hyper-parameters as the given one
		/// </summary>
		virtual bool equal_hyperparams(const ALayer& layer) const = 0;

		/// <summary>
		/// Returns sum of squared weight of the layer (needed, for example, to evaluate cost function when L2 regularization is involved)
		/// </summary>
		virtual Real squared_weights_sum() const = 0;
	};
}