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
#include "NLayer.h"
#include "../Math/ActivationFunction.h"
#include "../Math/CostFunction.h"
#include "../Math/Vector.h"
#include "../Math/Tensor.h"
#include "LayerHandle.h"
#include <msgpack.hpp>
#include <filesystem>

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
		std::vector<LayerHandle> _layers{};

		/// <summary>
		/// Returns sum of squared weight of all the layers
		/// </summary>
		Real squared_weights_sum() const;

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
		/// nonnegative integers. Each par of consecutive elements in the collection defines input (the left element) and
		/// output (the right element) dimensions of the corresponding neural layer. Thus the number of resulting neural
		/// layers is one less than the number of integers in the collection </param>
		/// <param name="activ_func_ids">Collection of activation function identifiers indicating the type of
		/// activation function to be used in each particular neural layer. The number of elements in this collection
		/// must either be equal to the number of neural layers (i.e. one less than the number of elements in "layer_dimensions") of be equal to "0".
		/// The latter case will be treated as if the collection was populated with SIGMOID identifiers.</param>
		Net(const std::vector<std::size_t>& layer_dimensions, const std::vector<ActivationFunctionId>& activ_func_ids = std::vector<ActivationFunctionId>());

		/// <summary>
		/// Instantiates net from the given script-string (see `to_script()` method below)
		/// </summary>
		/// <param name="script_str">Script-string, which can be output of `to_script()` method</param>
		Net(const std::string& script_str);

		/// <summary>
		/// Returns output of the neural network calculated for the given input
		/// </summary>
		Tensor act(const Tensor& input, std::vector<ALayer::AuxLearningData>* const aux_data_ptr = nullptr) const;

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
		void learn(const std::vector<Tensor>& training_items, const std::vector<Tensor>& reference_items,
			const std::size_t batch_size, const std::size_t epochs_count, const Real learning_rate, const CostFunctionId& cost_func_id,
			const Real& lambda = Real(0),
			const std::function<void(const std::size_t, const Real)>& epoch_callback = [](const auto epoch_id, const auto scaled_l2_reg_factor) {});

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
		std::size_t count_correct_answers(const std::vector<Tensor>& test_input, const std::vector<Tensor>& labels,
			const Real& min_answer_probability = Real(0)) const;

		/// <summary>
		/// Evaluates the given cost function for the given pair of input and reference collections
		/// </summary>
		Real evaluate_cost_function(const std::vector<Tensor>& test_input,
			const std::vector<Tensor>& reference_output, const CostFunctionId& cost_func_id, const Real l2_reg_factor) const;

		/// <summary>
		/// Method to append layer to the net
		/// </summary>
		/// <typeparam name="L">Later type</typeparam>
		/// <typeparam name="...Types">Types of arguments required by a constructor of type "L"</typeparam>
		/// <param name="...args">Actual arguments required by a constructor of type "L"</param>
		/// <returns>Output size of the appended layer</returns>
		template <class L, class... Types>
		Index3d append_layer(Types&&... args)
		{
			_layers.push_back(LayerHandle::make<L>(std::forward<Types>(args)...));
			return _layers.rbegin()->layer().out_size();
		}

		/// <summary>
		/// Logs the net into the given directory (will be created if it does not exist) 
		/// Logs of all the data are supposed to be done in a human-readable text format, since 
		/// the main purpose of the logging functionality is diagnostics
		/// </summary>
		/// <param name="directory">The directory of disk to log to</param>
		void log(const std::filesystem::path& directory) const;

		/// <summary>
		/// Encodes hyper-parameters of all the layers in a string-script which then can be used to instantiate 
		/// another instance of the net with the same set of hyper-parameters (see the constructor taking string argument)
		/// </summary>
		std::string to_script() const;

		/// <summary>
		/// Saves net as a script-like string to the given file
		/// </summary>
		/// <param name="scrypt_path">Path to the file to save script to</param>
		void save_script(const std::filesystem::path& scrypt_path) const;

		/// <summary>
		/// Instantiates net from the script in the given file on disk
		/// </summary>
		static Net load_script(const std::filesystem::path& scrypt_path);

		/// <summary>
		/// Returns "true" if the current and the given networks coincide in terms of hyper-parameters,
		/// i.e. layer types, their architecture etc.
		/// </summary>
		bool equal_hyperparams(const Net& net) const;

		/// <summary>
		/// Returns a human-readable description of the net through description of all its layers
		/// </summary>
		std::string to_string() const;
	};
}
