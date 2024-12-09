//Copyright (c) 2024 Denys Dragunov, dragunovdenis@gmail.com
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
#include "AMLayer.h"
#include "../Math/Matrix.h"
#include "../Math/LinAlg3d.h"
#include "../Math/ActivationFunction.h"
#include "../Math/InitializationStrategy.h"

namespace DeepLearning
{
	/// <summary>
	/// Recurrent multi-layer.
	/// </summary>
	template <class D = CpuDC>
	class RMLayer : public AMLayer<D>
	{
		/// <summary>
		/// Vector of bias coefficients.
		/// </summary>
		typename D::vector_t _biases{};

		/// <summary>
		/// Matrix of weights that applied to the input data.
		/// </summary>
		std::vector<typename D::matrix_t> _weights{2};

		/// <summary>
		/// Indices of "input" and "recursive" weight matrices in the collection of weights. 
		/// </summary>
		static constexpr int IN_W = 0;
		static constexpr int REC_W = 1;
		static constexpr int MSG_PACK_VER = 1;

		Index4d _in_size{};
		Index4d _out_size{};

		/// <summary>
		/// Access to the "input" weight matrix.
		/// </summary>
		typename D::matrix_t& in_weights();

		/// <summary>
		/// Access to the "input" weight matrix ("const" version).
		/// </summary>
		const typename D::matrix_t& in_weights() const;

		/// <summary>
		/// Access to the "recursive" weight matrix.
		/// </summary>
		typename D::matrix_t& rec_weights();

		/// <summary>
		/// Access to the "recursive" weight matrix ("const" version).
		/// </summary>
		const typename D::matrix_t& rec_weights() const;

		/// <summary>
		/// Recursion depth, i.e., how many recurrent iterations the layer does,
		///	which coincides with the dimensionality of an array of tensors
		///	that the layer receives as an input.
		/// </summary>
		long long& rec_depth();

		/// <summary>
		/// Const version of the corresponding method.
		/// </summary>
		const long long& rec_depth() const;

		/// <summary>
		/// Dimensionality of tensors in the output collection that the layer produces.
		/// </summary>
		Index3d& out_sub_dim();

		/// <summary>
		/// Const version of the corresponding method.
		/// </summary>
		/// <returns></returns>
		const Index3d& out_sub_dim() const;

		/// <summary>
		/// Pointer to an instance of activation function.
		/// </summary>
		std::shared_ptr<AFunction<typename D::tensor_t>> _func{};

		/// <summary>
		/// Identifier of activation function.
		/// </summary>
		ActivationFunctionId _func_id{ ActivationFunctionId::UNKNOWN };

		/// <summary>
		/// Returns reference to an instance of activation function.
		/// </summary>
		const AFunction<typename D::tensor_t>& get_func() const;

		/// <summary>
		/// Instantiates activation function based on the given ID.
		/// </summary>
		void set_func_id(const ActivationFunctionId func_id);

		/// <summary>
		/// Calculates value of activation function and in-place as well as its derivative if required.
		/// </summary>
		void apply_activation_function(typename D::tensor_t& in_out, typename D::tensor_t* const out_derivatives) const;

	public:

		/// <summary>
		/// Input dimensions of the layer.
		/// </summary>
		Index4d in_size() const override;

		/// <summary>
		/// Output dimensions of the layer.
		/// </summary>
		Index4d out_size() const override;

		/// <summary>
		/// Default constructor.
		/// </summary>
		RMLayer() = default;

		/// <summary>
		/// Constructor.
		/// </summary>
		/// <param name="rec_depth">Dimensionality of the input collection of
		/// tensors that the layer can deal with, coincides with the dimensionality
		/// of the output collection and with the recursion depth.</param>
		/// <param name="in_sub_dim">Linear size of each the tensors in the input collection.</param>
		/// <param name="out_sub_dim">3d shape of tensors in the output collection.</param>
		/// <param name="init_strategy">Strategy used to fill weight during the initialization phase.</param>
		/// <param name="func_id">Identifier of activation function.</param>
		RMLayer(const int rec_depth, const int in_sub_dim, const Index3d& out_sub_dim,
			const InitializationStrategy init_strategy, ActivationFunctionId func_id = ActivationFunctionId::TANH);

		/// <summary>
		/// Constructor.
		/// </summary>
		/// <param name="in_size">Dimensionality of the layer's input.</param>
		/// <param name="out_size">Dimensionality of the layer's output.</param>
		/// <param name="init_strategy">Strategy used to fill weight during the initialization phase.</param>
		/// <param name="func_id">Identifier of activation function.</param>
		RMLayer(const Index4d& in_size, const Index4d& out_size,
			const InitializationStrategy init_strategy, ActivationFunctionId func_id = ActivationFunctionId::TANH);

		/// <summary>
		/// Constructs an instance using the given set of the weights and biases.
		/// </summary>
		/// <param name="rec_depth">Recurrence depth.</param>
		/// <param name="out_sub_dim">Dimensions of a single output item.</param>
		/// <param name="in_w">"Input" weights.</param>
		/// <param name="r_w">"Recurrence" weights.</param>
		/// <param name="b">Biases.</param>
		/// <param name="func_id">Activation function</param>
		RMLayer(const int rec_depth, const Index3d& out_sub_dim, const typename D::matrix_t& in_w,
			const typename D::matrix_t& r_w, const typename D::vector_t& b,
			ActivationFunctionId func_id = ActivationFunctionId::TANH);

		/// <summary>
		/// Returns a container to be used in back-propagation procedure./>
		/// </summary>
		MLayerGradient<D> allocate_gradient_container() const override;

		/// <summary>
		/// Calculates result of the layer evaluated on the given <param name="input"/> data.
		/// </summary>
		/// <param name="input">Input data.</param>
		/// <param name="output">Container to receive the result.</param>
		/// <param name="trace_data">Pointer to data container to store "trace data",
		/// that later can be used during the backpropagation phase. Can be null,
		/// in which case no "trace date" will be stored.</param>
		void act(const IMLayerExchangeData<typename D::tensor_t>& input, IMLayerExchangeData<typename D::tensor_t>& output,
			IMLayerTraceData<D>* const trace_data) const override;

		/// <summary>
		/// Calculates derivatives with respect to all the parameters of the layer.
		/// </summary>
		/// <param name="out_grad">Derivatives of the cost function with respect
		/// to the output of the layer.</param>
		/// <param name="output">The output of the layer.</param>
		/// <param name="processing_data">Intermediate data prepared during the "act"
		/// phase that is needed to calculate the derivatives.</param>
		/// <param name="out_input_grad">A container that holds gradient of the layer's input upon the method's return.</param>
		/// <param name="out_layer_grad">A container that holds the layer's gradient upon the method's return.</param>
		/// <param name="evaluate_input_gradient">If "false" gradient of the layers input won't be calculated.</param>
		void backpropagate(const IMLayerExchangeData<typename D::tensor_t>& out_grad, const IMLayerExchangeData<typename D::tensor_t>& output,
			const IMLayerExchangeData<LayerData<D>>& processing_data, IMLayerExchangeData<typename D::tensor_t>& out_input_grad,
			MLayerGradient<D>& out_layer_grad, const bool evaluate_input_gradient = true) const override;

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const;

		/// <summary>
		/// Returns "true" if the current instance of the layer has the same set of hyper-parameters as the given one
		/// </summary>
		bool equal_hyperparams(const AMLayer<D>& layer) const override;

		/// <summary>
		/// Returns "true" if the given layer is (absolutely) equal to the current one
		/// </summary>
		bool equal(const AMLayer<D>& layer) const override;
	};
}

#include "MRLayer.inl"