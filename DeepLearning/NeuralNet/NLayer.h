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
#include "LayerTypeId.h"

namespace DeepLearning
{
	/// <summary>
	/// Representation of a single neural layer
	/// </summary>
	template <class D = CpuDC>
	class NLayer : public ALayer<D>
	{
	private:
		//Declare "friends" to be able to switch between the data contexts
		//(see 'to_host()' and 'to_device()' methods)
		friend class NLayer<CpuDC>;
		friend class NLayer<GpuDC>;

		/// <summary>
		/// Vector of bias coefficients of size _out_dim;
		/// </summary>
		typename D::vector_t _biases{};

		/// <summary>
		/// Matrix of weights of size _out_dim x _in_dim  
		/// </summary>
		typename D::matrix_t _weights{};

		/// <summary>
		/// Activation function id
		/// </summary>
		ActivationFunctionId _func_id = ActivationFunctionId::UNKNOWN;

		/// <summary>
		/// Initializes the layer according to the given set of parameters
		/// </summary>
		/// <param name="in_dim">Input dimension of the later</param>
		/// <param name="out_dim">Output dimension of the later</param>
		/// <param name="func_id">Identifier of the activation function of the layer</param>
		/// <param name="rand_low">Lower boundary for the random initialization of weight and biases</param>
		/// <param name="rand_high">Higher boundary for the random initialization of weights and biases</param>
		/// <param name="standard_init_for_weights">Applies "special" random initialization for the weights if "true"</param>
		void initialize(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id,
			const Real rand_low, const Real rand_high, const bool standard_init_for_weights);
	public:

		/// <summary>
		/// Layer type identifier
		/// </summary>
		static LayerTypeId ID() { return LayerTypeId::FULL; }

		MSGPACK_DEFINE(this->_keep_rate, _biases, _weights, _func_id);

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
		NLayer() = default;

		/// <summary>
		/// Constructor with random weights and biases
		/// </summary>
		NLayer(const std::size_t in_dim, const std::size_t out_dim, ActivationFunctionId func_id = ActivationFunctionId::SIGMOID,
			const Real rand_low = Real(-1), const Real rand_high = Real(1), const bool standard_init_for_weights = false,
			const Real keep_rate = Real(1.0));

		/// <summary>
		/// Constructor to instantiate layer from the given string of certain format
		/// </summary>
		NLayer(const std::string& str);

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		typename D::tensor_t act(const typename D::tensor_t& input, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr = nullptr) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		void act(const typename D::tensor_t& input, typename D::tensor_t& output, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr = nullptr) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		std::tuple<typename D::tensor_t, typename ALayer<D>::LayerGradient> backpropagate(const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
			const bool evaluate_input_gradient = true) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		void backpropagate(const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
		                   typename D::tensor_t& input_grad, typename ALayer<D>::LayerGradient& layer_grad, const bool evaluate_input_gradient = true) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		void update(const std::tuple<std::vector<typename D::tensor_t>, typename D::tensor_t>& weights_and_biases_increment, const Real& reg_factor) override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		void log(const std::filesystem::path& directory) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		std::string to_string() const override;

		/// <summary>
		/// Returns "true" if the current instance of the layer has the same set of hyper-parameters as the given one
		/// </summary>
		bool equal_hyperparams(const ALayer<D>& layer) const override;

		/// <summary>
		/// Returns "true" if the given layer is (absolutely) equal to the current one
		/// </summary>
		bool equal(const ALayer<D>& layer) const override;

		/// <summary>
		/// Encodes hyper-parameters of the layer in a string-script which then can be used to instantiate 
		/// another instance of the layer with the same set of hyper-parameters (see the constructor taking string argument)
		/// </summary>
		std::string to_script() const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		LayerTypeId get_type_id() const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		Real squared_weights_sum() const override;

		/// <summary>
		/// Converts the given instance to the one working within the "cpu data context"
		/// </summary>
		NLayer<CpuDC> to_host() const;

		/// <summary>
		/// Converts the given instance to the one working within the "gpu data context" (CUDA "device" memory)
		/// </summary>
		NLayer<GpuDC> to_device() const;
	};
}
