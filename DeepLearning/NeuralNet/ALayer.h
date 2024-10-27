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
#include "CumulativeGradient.h"
#include <filesystem>
#include <random>
#include <string>
#include "LayerTypeId.h"
#include "LayerGradient.h"
#include "../Math/ActivationFunction.h"

namespace DeepLearning
{
	/// <summary>
	/// An abstract neural net layer
	/// </summary>
	template <class D>
	class ALayer
	{
		/// <summary>
		/// An inverted drop-out mask, i.e. a mask that has "ones" on the positions with the indices of elements
		/// that we want to keep and "zeros" on the positions with the indices of elements that we want to "drop"
		/// </summary>
		mutable typename D::vector_t _keep_mask{};

		/// <summary>
		/// Auxiliary collection used to generate random keep masks
		/// </summary>
		mutable typename D::template index_array_t<int> _keep_mask_aux_collection{};

		/// <summary>
		///	Default value of the "keep rate" parameter
		/// </summary>
		static constexpr Real DefaultKeepRate = static_cast<Real>(1);

		std::shared_ptr<AFunction<typename D::tensor_t>> _func{};
		ActivationFunctionId _func_id = ActivationFunctionId::UNKNOWN;

		int _msg_pack_version = 1;

		/// <summary>
		/// One minus dropout rate
		/// </summary>
		Real _keep_rate = DefaultKeepRate;

		/// <summary>
		/// Json keys
		/// </summary>
		static constexpr auto json_keep_id() { return "Keep"; };
		static constexpr auto json_activation_id() { return "Activation"; };

	protected:

		/// <summary>
		/// Returns reference to random generator.
		/// </summary>
		static std::mt19937& ran_gen();

		/// <summary>
		/// Returns constant reference to the activation function
		/// </summary>
		const AFunction<typename D::tensor_t>& get_func() const;

		/// <summary>
		/// Method to set id of the activation function. Instantiates the corresponding function.
		/// </summary>
		void set_func_id(const ActivationFunctionId func_id);

		/// <summary>
		/// Sets the value of "keep rate".
		/// </summary>
		void set_keep_rate(const Real keep_rate);

	public:

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const
		{
			msgpack::type::make_define_array(_msg_pack_version, _keep_rate, _func_id).msgpack_pack(msgpack_pk);
		}

		/// <summary>
		/// Getter for the "keep rate" property
		/// </summary>
		Real get_keep_rate() const;

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="keep_rate">One minus "dropout" rate, is supposed to be in (0, 1]. Keep rate "1.0" means that
		/// no dropout will be used</param>
		/// <param name="func_id">ID of the activation function</param>
		ALayer(const Real keep_rate, const ActivationFunctionId func_id = ActivationFunctionId::UNKNOWN);

		/// <summary>
		/// Default constructor
		/// </summary>
		ALayer() = default;

		/// <summary>
		/// Instantiation from the given script
		/// </summary>
		ALayer(const std::string& script);

		/// <summary>
		/// Initializes (or re-initializes) the mask that will be used to perform training with "drop-put" regularization
		/// In fact it initializes an "inversion" of the drop-out mask, that is the "keep" mask
		/// (as a rule this should be called by the training subroutine before each mini-batch)
		/// </summary>
		void SetUpDropoutMask() const;

		/// <summary>
		/// Method to free resources that has been allocated to do the "drop-out" regularization
		/// (should be called by the training subroutine when the training is over)
		/// </summary>
		void DisposeDropoutMask() const;

		/// <summary>
		///	Applies dropout to the given tensor that is supposed to be an input for the current layer
		/// </summary>
		/// <param name="input">The input tensor for the current layer</param>
		/// <param name="trainingMode">Mode flag. Should be set to "true" when training and "false" when inferring.</param>
		void ApplyDropout(typename D::tensor_t& input, const bool trainingMode) const;

		/// <summary>
		/// Auxiliary data to perform learning on a level of single convolution layer
		/// </summary>
		struct AuxLearningData
		{
			/// <summary>
			/// Container to store input of a convolution layer
			/// </summary>
			typename D::tensor_t Input{};

			/// <summary>
			/// Container to store derivatives of the activation function
			/// </summary>
			typename D::tensor_t Derivatives{};

			/// <summary>
			/// Place holder for index mappings
			/// </summary>
			typename D::template index_array_t<int> IndexMapping{};
		};

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
		/// <param name="input">Input "signal"</param>
		/// <param name="aux_learning_data_ptr">Pointer to the auxiliary data structure that should be provided during the training (learning) process</param>
		/// <returns>Output signal</returns>
		virtual typename D::tensor_t act(const typename D::tensor_t& input, AuxLearningData* const aux_learning_data_ptr = nullptr) const = 0;

		/// <summary>
		/// Makes a forward pass for the given input and places the result to the given ("output") container
		/// </summary>
		/// <param name="input">Input "signal"</param>
		/// <param name="output">Place-holder for the result</param>
		/// <param name="aux_learning_data_ptr">Pointer to the auxiliary data structure that should be provided during the training (learning) process</param>
		virtual void act(const typename D::tensor_t& input, typename D::tensor_t& output, AuxLearningData* const aux_learning_data_ptr = nullptr) const = 0;

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
		virtual std::tuple<typename D::tensor_t, LayerGradient<D>> backpropagate(const typename D::tensor_t& deltas, const AuxLearningData& aux_learning_data,
			const bool evaluate_input_gradient = true) const = 0;

		/// <summary>
		/// Performs the back-propagation
		/// </summary>
		/// <param name="deltas">Derivatives of the cost function with respect to the output of the current layer</param>
		/// <param name="aux_learning_data">Auxiliary learning data that should be obtained from the corresponding
		/// "forward" pass (see method "act") </param>
		/// <param name="input_grad">Place-holder for the gradient with respect to the input data</param>
		/// <param name="layer_grad">Place-holder for the gradient with respect to the parameters of the layer itself</param>
		/// <param name="evaluate_input_gradient">Determines whether the gradient with respect to the
		/// input data will be actually evaluated.
		/// The evaluation is redundant for the very first layer of the net</param>
		/// <param name="gradient_scale_factor">Scale factor to be applied to the content of
		/// `kernel_grad` before adding the gradient value to it.</param>
		virtual void backpropagate(const typename D::tensor_t& deltas, const AuxLearningData& aux_learning_data,
			typename D::tensor_t& input_grad, LayerGradient<D>& layer_grad,
			const bool evaluate_input_gradient = true, const Real gradient_scale_factor = static_cast<Real>(0)) const = 0;

		/// <summary>
		/// Resizes the given container so that it matches the size of the layer's gradient.
		/// </summary>
		virtual void allocate(LayerGradient<D>& gradient_container, bool fill_zeros) const = 0;

		/// <summary>
		/// Adds given increments multiplied by the "learning rate" to the weights and biases respectively
		/// </summary>
		/// <param name="gradient">Increment for weights and biases</param>
		/// <param name="l_rate">Learning rate</param>
		/// <param name="reg_factor">Regularization factor, that (if non-zero) 
		/// results in term "reg_factor*w_i" being added to each weight "w_i" </param>
		virtual void update(const LayerGradient<D>& gradient, const Real& l_rate, const Real& reg_factor) = 0;

		/// <summary>
		/// Returns zero initialized instance of cumulative gradient suitable for the current instance of the layer
		/// </summary>
		[[nodiscard]] virtual CumulativeGradient<D> init_cumulative_gradient() const
		{
			return CumulativeGradient<D>(weight_tensor_size(), out_size());
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
		virtual bool equal_hyperparams(const ALayer<D>& layer) const = 0;

		/// <summary>
		/// Returns "true" if the given layer is (absolutely) equal to the current one
		/// </summary>
		virtual bool equal(const ALayer<D>& layer) const = 0;

		/// <summary>
		/// Returns sum of squared weight of the layer (needed, for example, to evaluate cost function when L2 regularization is involved)
		/// </summary>
		virtual Real squared_weights_sum() const = 0;

		/// <summary>
		/// Sets all the "trainable" parameters (weights and biases) to zero
		/// </summary>
		virtual void reset() = 0;

		/// <summary>
		/// Resets random generator with the given seed.
		/// </summary>
		static void reset_random_generator(const unsigned seed);

		/// <summary>
		/// Resets random generator with the std::random_device generated seed.
		/// </summary>
		static void reset_random_generator();
	};
}

#include "ALayer.inl"