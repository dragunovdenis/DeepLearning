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
#include "CostsAndCorrectAnswers.h"
#include "LayerHandle.h"
#include <msgpack.hpp>
#include <filesystem>

namespace DeepLearning
{
	/// <summary>
	/// Class representing a neural network consisting of neural layers
	/// </summary>
	template <class D = CpuDC>
	class Net
	{
		/// <summary>
		/// Returns reference to a random number generator.
		/// </summary>
		static std::mt19937& ran_gen();

		/// <summary>
		/// Applies random permutation to the given collection of indices
		/// </summary>
		static void apply_random_permutation(std::vector<std::size_t>& indices);

		/// <summary>
		/// Returns collection containing "elements_count" integer
		/// indices starting from "0" all the way to "elements_count - 1"
		/// </summary>
		static std::vector<std::size_t> get_indices(const std::size_t elements_count);

		/// <summary>
		/// Returns collection of the gradient collectors that is "compatible" with the given collection of neural layers
		/// </summary>
		static std::vector<CumulativeGradient<D>> init_gradient_collectors(const std::vector<LayerHandle<D>>& layers);

		/// <summary>
		/// Resets all the collectors in the given collection
		/// </summary>
		static void reset_gradient_collectors(std::vector<CumulativeGradient<D>>& collectors);

		/// <summary>
		/// Allocates gradient container for multi-thread computations
		/// </summary>
		static void allocate_per_thread(const Net<D>& net, std::vector<std::vector<LayerGradient<D>>>& per_thread_gradient_container);

		/// <summary>
		/// Layers of neurons
		/// </summary>
		std::vector<LayerHandle<D>> _layers{};

		/// <summary>
		/// Returns sum of squared weight of all the layers
		/// </summary>
		Real squared_weights_sum() const;

		/// <summary>
		/// Auxiliary data structure used for more efficient memory usage when evaluating networs
		/// </summary>
		struct InOutData
		{
			/// <summary>
			/// Input for a layer
			/// </summary>
			typename D::tensor_t In{};

			/// <summary>
			/// Output of a layer
			/// </summary>
			typename D::tensor_t Out{};

			/// <summary>
			/// Swaps input ans output fields
			/// </summary>
			void swap()
			{
				std::swap(In, Out);
			}
		};

	public:
		/// <summary>
		/// Data structure that represents auxiliary resources needed to do inferring/training of the neural net 
		/// </summary>
		class Context
		{
			friend class Net;//keep everything private and visible only for the Net class
			/// <summary>
			/// Auxiliary resources used in the calculation of the neural net's gradient 
			/// </summary>
			std::vector<typename ALayer<D>::AuxLearningData> gradient_cache;
			
			/// <summary>
			/// Auxiliary resources used in the calculation of the neural net's value 
			/// </summary>
			InOutData value_cache;

		public:
			/// <summary>
			/// Read-only access to the "out" of the value cache
			/// </summary>
			const typename D::tensor_t& get_out() const;

			/// <summary>
			/// Default constructor
			/// </summary>
			Context() = default;

			/// <summary>
			/// Constructor allocating resources for the given number of layers
			/// </summary>
			Context(const std::size_t layers_count):gradient_cache(layers_count){}
		};

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
		/// Tries load a net from the given script-string (for example, an output of `to_script()` method)
		/// Throws exception with the corresponding message if fails
		/// </summary>
		/// <param name="script">Script-string</param>
		void try_load_from_script(const std::string& script);

		/// <summary>
		/// Tries to load a net from a string-script in the given file
		/// Throws exception with the corresponding message if fails
		/// </summary>
		/// <param name="file_path">Path to file with a string-script (generated, for example by `to_script()` method)</param>
		void try_load_from_script_file(const std::filesystem::path& file_path);

		/// <summary>
		/// Returns output of the neural network calculated for the given input
		/// </summary>
		typename D::tensor_t act(const typename D::tensor_t& input) const;

		/// <summary>
		/// Evaluates network at the given input
		/// </summary>
		/// <param name="input">Input tensor</param>
		/// <param name="context">Computation context (memory that can be allocated once and then re-used in many computations).
		/// Context as such is not thread-safe</param>
		/// <param name="calc_gradient_cache">If "true", gradient cache will be calculated during the method invocation and stored to the context.
		/// The memory for the gradient cache should be allocated by the caller.</param>
		void act(const typename D::tensor_t& input, Context& context, const bool calc_gradient_cache) const;

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
		/// <param name="single_threaded">Determines whether learning should be run in a single-threaded
		/// way to ensure determinism (presumably for testing purposes)</param>
		void learn(const std::vector<typename D::tensor_t>& training_items, const std::vector<typename D::tensor_t>& reference_items,
			const std::size_t batch_size, const std::size_t epochs_count, const Real learning_rate, const CostFunctionId& cost_func_id,
			const Real& lambda = Real(0),
			const std::function<void(const std::size_t, const Real)>& epoch_callback =
			[](const auto epoch_id, const auto scaled_l2_reg_factor) {}, const bool single_threaded = false);

		/// <summary>
		/// Specific implementation of the corresponding general function which
		/// performs learning on a single labeled item (single epoch, "batch size" equal to one)
		/// Such an implementation can be handy when running, for example,
		/// a semi-gradient-based approximation of a state-value function in reinforcement learning tasks.
		/// </summary>
		/// <param name="training_item">Training item</param>
		/// <param name="target_value">Label of the training item</param>
		/// <param name="learning_rate">Learning rate</param>
		/// <param name="cost_func_id">Identifier of the cost function to use int the learning process</param>
		/// <param name="lambda">Parameter of regularization</param>
		void learn(const typename D::tensor_t& training_item, const typename D::tensor_t& target_value,
		           const Real learning_rate, const CostFunctionId& cost_func_id, const Real& lambda = Real(0));

		/// <summary>
		/// Returns gradient of the given cost function with respect to the weights and biases of the neural net.
		/// Additionally to that returns the "value" of the network inferred at the given "item".
		/// </summary>
		/// <param name="item">An item at which the gradient should be evaluated</param>
		/// <param name="target_value">"Label" item that should be used in the cost function</param>
		/// <param name="cost_func_id">Id of the cost function</param>
		std::tuple<std::vector<LayerGradient<D>>, typename D::tensor_t> calc_gradient_and_value(
			const typename D::tensor_t& item, const typename D::tensor_t& target_value, const CostFunctionId& cost_func_id) const;

		/// <summary>
		/// Calculates gradient of the given cost function with respect to the weights and biases of the neural net.
		/// Additionally to that calculates the "value" of the network inferred at the given "item".
		/// </summary>
		/// <param name="item">An item at which the gradient should be evaluated</param>
		/// <param name="target_value">"Label" item that should be used in the cost function</param>
		/// <param name="cost_func_id">Id of the cost function</param>
		/// <param name="out_gradient">Output parameter the gradient is calculated into</param>
		/// <param name="out_value">Output parameter the value is calculated into</param>
		/// <param name="gradient_scale_factor">Scale factor to be applied to the content of the `out_gradient`
		/// container before adding the calculated gradient to it.</param>
		/// <param name="context">Calculation context, i.e. resources tht can be
		/// allocated once and then re-used in each call of the method. Serves optimization purposes</param>
		void calc_gradient_and_value(const typename D::tensor_t& item, const typename D::tensor_t& target_value,
			const CostFunctionId& cost_func_id, std::vector<LayerGradient<D>>& out_gradient, typename D::tensor_t& out_value,
			const Real gradient_scale_factor, Context& context) const;

		/// <summary>
		/// Updates weights and biases of all the layers with the given gradient
		/// according to the given learning rate and regularization factor
		/// </summary>
		/// <param name="gradient">Collection of gradient structures, one for each layer.</param>
		/// <param name="learning_rate">Learning rate</param>
		/// <param name="lambda">Regularization factor</param>
		void update(const std::vector<typename DeepLearning::LayerGradient<D>>& gradient, const Real learning_rate, const Real& lambda);

		/// <summary>
		/// Evaluates number of "correct answers" for the given collection of the
		/// input data with respect to the given collection of reference labels 
		/// It is assumed that the labels are in effect zero vectors with single positive element (defining the "correct" class)
		/// (i.e. the net is trained to solve classification problems)
		/// </summary>
		/// <param name="test_input">Input data</param>
		/// <param name="labels">Labels for the given input data</param>
		/// from the neural net should have in order to be considered as a "valid"</param>
		std::size_t count_correct_answers(const std::vector<typename D::tensor_t>& test_input, const std::vector<typename D::tensor_t>& labels) const;

		/// <summary>
		/// Evaluates the given cost function for the given pair of input and reference collections as well as the percentage of correct answers 
		/// (ratio of correct answers to all the answers)
		/// </summary>
		CostAndCorrectAnswers evaluate_cost_function_and_correct_answers(const std::vector<typename D::tensor_t>& test_input,
			const std::vector<typename D::tensor_t>& reference_output, const CostFunctionId& cost_func_id, const Real l2_reg_factor) const;

		/// <summary>
		/// Method to append layer to the net
		/// </summary>
		/// <typeparam name="L">Later type</typeparam>
		/// <typeparam name="...Types">Types of arguments required by a constructor of type "L"</typeparam>
		/// <param name="args">Actual arguments required by a constructor of type "L"</param>
		/// <returns>Output size of the appended layer</returns>
		template <template<class> class L, class... Types>
		Index3d append_layer(Types&&... args)
		{
			_layers.emplace_back(LayerHandle<D>::template make<L<D>>(std::forward<Types>(args)...));
			return _layers.rbegin()->layer().out_size();
		}

		/// <summary>
		/// Allocates memory needed to store gradient of all the parameters of the neural net (in layer-wise manner).
		/// </summary>
		void allocate(std::vector<LayerGradient<D>>& gradient_container, bool fill_zeros) const;

		/// <summary>
		///	Read-only access to the layers
		/// </summary>
		const ALayer<D>& operator [](const std::size_t& id) const;

		/// <summary>
		/// Access to the layers
		/// </summary>
		ALayer<D>& operator [](const std::size_t& id);

		/// <summary>
		///	Returns number of layers in the net
		/// </summary>
		[[nodiscard]] std::size_t layers_count() const;

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
		[[nodiscard]] std::string to_script() const;

		/// <summary>
		/// Saves net as a script-like string to the given file
		/// </summary>
		/// <param name="script_path">Path to the file to save script to</param>
		void save_script(const std::filesystem::path& script_path) const;

		/// <summary>
		/// Returns "true" if the current and the given networks coincide in terms of hyper-parameters,
		/// i.e. layer types, their architecture etc.
		/// </summary>
		bool equal_hyperparams(const Net& net) const;

		/// <summary>
		///	Returns "true" if the given net is (absolutely) equal to the current one
		/// </summary>
		bool equal(const Net& net) const;

		/// <summary>
		/// Returns a human-readable description of the net through description of all its layers
		/// </summary>
		[[nodiscard]] std::string to_string() const;

		/// <summary>
		///	Returns dimensions of the input data item (negative if the network does not have layers yet) 
		/// </summary>
		[[nodiscard]] Index3d in_size() const;

		/// <summary>
		///	Returns dimensions of the output data item (negative if the network does not have layers yet)
		/// </summary>
		[[nodiscard]] Index3d out_size() const;

		/// <summary>
		/// Returns collection of input dimensions of all the neural layers
		/// in the order that corresponds to the order of layers and,
		/// additionally, the output dimension of the last layer at the end of the returned collection
		/// </summary>
		[[nodiscard]] std::vector<Index3d> get_dimensions() const;

		/// <summary>
		/// Sets all the "trainable" parameters (weights and biases) to zero
		/// </summary>
		void reset();

		/// <summary>
		/// Resets random generator with the given seed.
		/// </summary>
		static void reset_random_generator(const unsigned seed);

		/// <summary>
		/// Resets random generator with std::random_device generated seed
		/// </summary>
		static void reset_random_generator();

		/// <summary>
		/// Serializes current instance to the given file (in messagepack format)
		/// </summary>
		void save_to_file(const std::filesystem::path& file_name) const;

		/// <summary>
		/// Loads instance of the net from the given file (in messagepack format)
		/// </summary>
		static Net<D> load_from_file(const std::filesystem::path& file_name);
	};
}

#include "Net.inl"