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
		/// <summary>
		/// Layers of neurons
		/// </summary>
		std::vector<NeuralLayer> _layers{};

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
		DenseVector act(const DenseVector& input, std::vector<NeuralLayer::AuxLearningData>* const aux_data_ptr = nullptr) const;

		/// <summary>
		/// A method that performs training of the neural net based on the given input data with references
		/// </summary>
		/// <param name="training_items">Collection of training items</param>
		/// <param name="reference_items">Collection of references (labels). One for each training item.</param>
		/// <param name="batch_size">Number of elements in a batch (stochastic gradient descent method)</param>
		/// <param name="epochs_count">Number of epochs to perform</param>
		/// <param name="cost_func_id">Identifier of the cost function to use in the training process</param>
		/// <param name="learning_rate">The learning rate (expected to be positive)</param>
		/// <param name="lambda">Regularization factor, determining the regularization term that will be added to the cost function : lambda/(2*n)*\sum w_{i},
		/// where n is the number of training items, w_{i} --- weights.</param>
		/// <param name="epoch_callback">Callback method that will be called after each learning epoch.
		/// It is supposed to serve diagnostic evaluation purposes.</param>
		void learn(const std::vector<DenseVector>& training_items, const std::vector<DenseVector>& reference_items,
			const std::size_t batch_size, const std::size_t epochs_count, const Real learning_rate, const CostFunctionId& cost_func_id,
			const Real& lambda = Real(0),
			const std::function<void(std::size_t)>& epoch_callback = [](const auto epoch_id) {});

		/// <summary>
		/// Evaluates number of "correct answers" for the given collection of the
		/// input data with respect to the given collection of reference labels 
		/// It is assumed that the labels are in effect zero vectors with single positive element (defining the "correct" class)
		/// (i.e. the net is trained to solve classification problems)
		/// </summary>
		/// <param name="test_items">Input data</param>
		/// <param name="labels">Labels for the given input data</param>
		/// <param name="min_answer_probability">Minimal "probability" that the answer
		/// from the neural net should have in order to be considered as a "valid"</param>
		std::size_t count_correct_answers(const std::vector<DenseVector>& test_input, const std::vector<DenseVector>& labels,
			const Real& min_answer_probability = Real(0)) const;

		/// <summary>
		/// Evaluates the given cost function for the given pair of input and reference collections
		/// </summary>
		Real evaluate_cost_function(const std::vector<DenseVector>& test_input,
			const std::vector<DenseVector>& reference_output, const CostFunctionId& cost_func_id) const;
	};
}