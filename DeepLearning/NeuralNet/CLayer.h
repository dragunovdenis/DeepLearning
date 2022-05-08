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
#include "ALayer.h"
#include "../Math/ActivationFunction.h"
#include "../Math/Tensor.h"
#include "../Math/LinAlg2d.h"
#include <vector>
#include <msgpack.hpp>
#include "LayerTypeId.h"

namespace DeepLearning
{
	/// <summary>
	/// "Convolution" layer
	/// </summary>
	class CLayer : public ALayer
	{
		Index3d _in_size{};
		Index3d _weight_tensor_size{};
		Index3d _paddings{};
		Index3d _strides{};
		ActivationFunctionId _func_id = ActivationFunctionId::UNKNOWN;

		/// <summary>
		/// Biases
		/// </summary>
		Tensor _biases{};

		/// <summary>
		/// Convolution filters
		/// </summary>
		std::vector<Tensor> _filters{};

	public:

		/// <summary>
		/// Layer type identifier
		/// </summary>
		static LayerTypeId ID() { return LayerTypeId::CONVOLUTION; }

		MSGPACK_DEFINE(_in_size, _weight_tensor_size, _paddings, _strides, _func_id, _biases, _filters);

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual Index3d in_size() const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual Index3d out_size() const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual Index3d weight_tensor_size() const override;

		/// <summary>
		/// Default constructor
		/// </summary>
		CLayer() {}

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="in_size">Size of the input tensor</param>
		/// <param name="filter_window_size">Size of a 2d window (in a channel) for the filters to operate</param>
		/// <param name="filters_count">Number of filters. Defines number of channel in the output tensor of the layer</param>
		/// <param name="paddings">Zero paddings to be applied to the input tensor before filters are applied</param>
		/// <param name="strides">Define the minimal movement of the filter window (in 3 dimensions) when convolution is going on</param>
		/// <param name="funcId">Id of the activation function to be used in the layer</param>
		CLayer(const Index3d& in_size, const Index2d& filter_window_size,
			const std::size_t& filters_count, const Index3d& paddings, const Index3d& strides, const ActivationFunctionId func_id);

		/// <summary>
		/// See description in the base class
		/// </summary>
		Tensor act(const Tensor& input, AuxLearningData* const aux_learning_data_ptr = nullptr) const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual std::tuple<Tensor, LayerGradient> backpropagate(const Tensor& deltas, const AuxLearningData& aux_learning_data,
			const bool evaluate_input_gradient = true) const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual void update(const std::tuple<std::vector<Tensor>, Tensor>& weights_and_biases_increment, const Real& reg_factor) override;
	};

}
