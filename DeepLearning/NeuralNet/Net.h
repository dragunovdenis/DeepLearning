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
#include "../defs.h"
#include "NeuralLayer.h"
#include "CummulativeGradient.h"
#include "../Math/ActivationFunction.h"
#include "../Math/CostFunction.h"
#include "../Math/DenseVector.h"
#include <msgpack.hpp>

namespace DeepLearning
{
	/// <summary>
	/// Class representing a neural network consisting of neural layers
	/// </summary>
	class Net
	{
		std::vector<NeuralLayer> _layers{};

		/// <summary>
		/// Toggles on/of the learning mode for all the layers of the net
		/// </summary>
		void SetLearningMode(const bool do_learning);

	public:

		MSGPACK_DEFINE(_layers);

		/// <summary>
		/// Default constructor (for the message-pack functionality to work)
		/// </summary>
		Net() = default;

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="layer_dimensions">A collection of
		/// nonnegative integers. Each par of consecutive elements in the collection defines input (left element) and
		/// output (right element) dimensions of the corresponding neural layer. Thus the number of resulting neural
		/// layers is one less than the number of integers in the collection </param>
		Net(const std::vector<std::size_t>& layer_dimensions, const ActivationFunctionId& activ_func_id = ActivationFunctionId::SIGMOID);

		/// <summary>
		/// Returns output of the neural network calculated for the given input
		/// </summary>
		DenseVector act(const DenseVector& input) const;

		/// <summary>
		/// A method that performs training of the neural net based on the given input data with references
		/// </summary>
		/// <param name="training_items">Collection of training items</param>
		/// <param name="reference_items">Collection of references (labels). One for each training item.</param>
		/// <param name="batch_size">Number of elements in a batch (stochastic gradient descent method)</param>
		/// <param name="epochs_count">Number of epochs to perform</param>
		/// <param name="cost_func_id">Identifier of the cost function to use in the training process</param>
		/// <param name="learning_rate">The learning rate (expected to be positive)</param>
		/// <returns>A collection of values of the given cost function evaluated on the given training data after each epoch</returns>
		std::vector<Real> learn(const std::vector<DenseVector>& training_items, const std::vector<DenseVector>& reference_items,
			const std::size_t batch_size, const std::size_t epochs_count, const Real learning_rate, const CostFunctionId& cost_func_id);
	};
}
