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
#include "ALayer.h"
#include "../Math/PoolOperator.h"
#include <msgpack.hpp>
#include "LayerTypeId.h"
#include "../Math/LinAlg2d.h"

namespace DeepLearning
{
	/// <summary>
	/// "Pooling" layer
	/// </summary>
	template <class D = CpuDC>
	class PLayer : public ALayer<D>
	{
		Index3d _in_size{};
		Index3d _pool_window_size{};
		Index3d _strides{};

		PoolTypeId _pool_operator_id = PoolTypeId::UNKNOWN;

		/// <summary>
		/// Initializes the layer according to the given set of parameters
		/// </summary>
		/// <param name="in_size">Input dimension of the layer</param>
		/// <param name="pool_window_size">Size of 2d pooling window</param>
		/// <param name="pool_operator_id">Identifier of the pooling operation to be used by the layer</param>
		void initialize(const Index3d& in_size, const Index2d& pool_window_size, const PoolTypeId pool_operator_id);

		int _msg_pack_version = 1;

	public:

		/// <summary>
		/// Layer type identifier
		/// </summary>
		static LayerTypeId ID() { return LayerTypeId::PULL; }

		//MSGPACK_DEFINE(this->_keep_rate, _in_size, _pool_window_size, _strides, _pool_operator_id);

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
			msgpack::type::make_define_array(_msg_pack_version, MSGPACK_BASE(ALayer<D>),
				_in_size, _pool_window_size, _strides, _pool_operator_id).msgpack_pack(msgpack_pk);
		}

		/// <summary>
		/// Default constructor
		/// </summary>
		PLayer() = default;

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="in_size">Size of input tensor for the layer</param>
		/// <param name="pool_window_size">2d window of the pool operator</param>
		/// <param name="pool_operator_id">Identifier of the pooling operator to be used</param>
		/// <param name="keep_rate">One minus "dropout" rate</param>
		PLayer(const Index3d& in_size, const Index2d& pool_window_size, const PoolTypeId pool_operator_id,
			const Real keep_rate = Real(1.0));

		/// <summary>
		/// Constructor to instantiate layer from the given string of certain format
		/// </summary>
		/// <param name="str"></param>
		PLayer(const std::string& str, const Index3d& default_in_size = Index3d::zero());

		/// <summary>
		/// Size of the layer's input
		/// </summary>
		Index3d in_size() const override;

		/// <summary>
		/// Size of the layer's output
		/// </summary>
		Index3d out_size() const override;

		/// <summary>
		/// Returns size of pooling operator window
		/// </summary>
		Index3d weight_tensor_size() const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		typename D::tensor_t act(const typename D::tensor_t& input, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr = nullptr) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		void act(const typename D::tensor_t& input, typename D::tensor_t& output, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr = nullptr) const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		std::tuple<typename D::tensor_t, LayerGradient<D>> backpropagate(
			const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
			const bool evaluate_input_gradient = true) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		void backpropagate(const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
			typename D::tensor_t& input_grad, LayerGradient<D>& layer_grad, const bool evaluate_input_gradient = true,
			const Real gradient_scale_factor = static_cast<Real>(0)) const override;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		void allocate(LayerGradient<D>& gradient_container, bool fill_zeros) const override;

		/// <summary>
		/// For the "pooling" layer this method does nothing except a sanity check that the input increments are empty
		/// </summary>
		void update(const LayerGradient<D>& gradient, const Real& l_rate, const Real& reg_factor) override;

		/// <summary>
		/// Returns zero initialized instance of cumulative gradient suitable for the current instance of the layer
		/// </summary>
		CummulativeGradient<D> init_cumulative_gradient() const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		void log(const std::filesystem::path& directory) const override {/*do nothing since this layer does not have a "state" to log anything*/ }

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
		PLayer<CpuDC> to_host() const;

		/// <summary>
		/// Converts the given instance to the one working within the "gpu data context" (CUDA "device" memory)
		/// </summary>
		PLayer<GpuDC> to_device() const;

		/// <summary>
		/// Sets all the "trainable" parameters (weights and biases) to zero
		/// </summary>
		void reset() override;
	};
}
