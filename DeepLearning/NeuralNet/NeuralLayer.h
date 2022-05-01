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
#include "../Math/Vector.h"
#include "../Math/Matrix.h"
#include "../Math/Tensor.h"
#include "../Math/LinAlg3d.h"
#include "ALayer.h"
#include <vector>
#include "../Math/ActivationFunction.h"
#include "CummulativeGradient.h"
#include <msgpack.hpp>

namespace DeepLearning
{
	/// <summary>
	/// Representation of a single neural layer
	/// </summary>
	class NeuralLayer : public ALayer
	{
	private:
		/// <summary>
		/// Vector of bias coefficients of size _out_dim;
		/// </summary>
		Vector _biases{};

		/// <summary>
		/// Matrix of weights of size _out_dim x _in_dim  
		/// </summary>
		Matrix _weights{};

		/// <summary>
		/// Activation function id
		/// </summary>
		ActivationFunctionId _func_id = ActivationFunctionId::UNKNOWN;

	public:

		MSGPACK_DEFINE(_biases, _weights, _func_id);

		/// <summary>
		/// Dimensionality of the layer's input
		/// </summary>
		Index3d in_size() const override;

		/// <summary>
		/// Dimensionality of the layer's output
		/// </summary>
		Index3d out_size() const override;

		/// <summary>
		/// Returns size of the weights matrix (tensor)
		/// </summary>
		Index3d weight_tensor_size() const override;

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
		NeuralLayer(const Matrix& weights, const Vector& biases, ActivationFunctionId func_id = ActivationFunctionId::SIGMOID);

		/// <summary>
		/// Copy constructor
		/// </summary>
		NeuralLayer(const NeuralLayer& anotherLayer);

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		Tensor act(const Tensor& input, AuxLearningData* const aux_learning_data_ptr = nullptr) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		std::tuple<Tensor, LayerGradient> backpropagate(const Tensor& deltas, const AuxLearningData& aux_learning_data,
			const bool evaluate_input_gradient = true) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		void update(const std::tuple<std::vector<Tensor>, Tensor>& weights_and_biases_increment, const Real& reg_factor) override;
	};

}
