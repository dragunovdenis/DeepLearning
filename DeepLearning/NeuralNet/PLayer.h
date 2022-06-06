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
	class PLayer : public ALayer
	{
		Index3d _in_size{};
		Index3d _pool_window_size{};
		const Index3d _paddings{}; //always use zero paddings
		Index3d _strides{};

		PoolTypeId _pool_operator_id = PoolTypeId::UNKNOWN;

		/// <summary>
		/// Initializes the layer according to the given set of parameters
		/// </summary>
		/// <param name="in_size">Input dimension of the layer</param>
		/// <param name="pool_window_size">Size of 2d pooling window</param>
		/// <param name="pool_operator_id">Identifier of the pooling operation to be used by the layer</param>
		void initialize(const Index3d& in_size, const Index2d& pool_window_size, const PoolTypeId pool_operator_id);

	public:

		/// <summary>
		/// Layer type identifier
		/// </summary>
		static LayerTypeId ID() { return LayerTypeId::PULL; }

		MSGPACK_DEFINE(_in_size, _pool_window_size, _strides, _pool_operator_id);

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
		PLayer(const Index3d& in_size, const Index2d& pool_window_size, const PoolTypeId pool_operator_id);

		/// <summary>
		/// Constructor to instantiate layer from the given string of certain format
		/// </summary>
		/// <param name="str"></param>
		PLayer(const std::string& str);

		/// <summary>
		/// Size of the layer's input
		/// </summary>
		virtual Index3d in_size() const override;

		/// <summary>
		/// Size of the layer's output
		/// </summary>
		virtual Index3d out_size() const override;

		/// <summary>
		/// Returns size of pooling operator window
		/// </summary>
		virtual Index3d weight_tensor_size() const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual Tensor act(const Tensor& input, AuxLearningData* const aux_learning_data_ptr = nullptr) const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual std::tuple<Tensor, LayerGradient> backpropagate(const Tensor& deltas, const AuxLearningData& aux_learning_data,
			const bool evaluate_input_gradient = true) const override;

		/// <summary>
		/// For the "pooling" layer this method does nothing except a sanity check that the input increments are empty
		/// </summary>
		virtual void update(const std::tuple<std::vector<Tensor>, Tensor>& weights_and_biases_increment, const Real& reg_factor) override;

		/// <summary>
		/// Returns zero initialized instance of cumulative gradient suitable for the current instance of the layer
		/// </summary>
		virtual CummulativeGradient init_cumulative_gradient() const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual void log(const std::filesystem::path& directory) const {/*do nothing since this layer does not have a "state" to log anything*/ }

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		virtual std::string to_string() const override;

		/// <summary>
		/// Returns "true" if the current instance of the layer has the same set of hyper-parameters as the given one
		/// </summary>
		bool equal_hyperparams(const ALayer& layer) const override;

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
	};
}
