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
	/// Convolutional layer
	/// </summary>
	template <class D>
	class CLayer : public ALayer<D>
	{
		Index3d _in_size{};
		Index3d _weight_tensor_size{};
		Index3d _paddings{};
		Index3d _strides{};
		ActivationFunctionId _func_id = ActivationFunctionId::UNKNOWN;

		//Declare "friends" to be able to switch between the data contexts
		//(see 'to_host()' and 'to_device()' methods)
		friend class CLayer<CpuDC>;
		friend class CLayer<GpuDC>;

		/// <summary>
		/// Copies all the field from the source to destination instance accept those that might
		/// require "host-to-device" or "device-to-host" copy operations
		/// </summary>
		template <class D1, class D2>
		static void partial_copy(const CLayer<D1>& src, CLayer<D2>& dest);

		/// <summary>
		/// Biases
		/// </summary>
		typename D::tensor_t _biases{};

		/// <summary>
		/// Convolution filters
		/// </summary>
		std::vector<typename D::tensor_t> _filters{};

		/// <summary>
		/// Initializes the layer according to the given set of parameters
		/// </summary>
		/// <param name="in_size">Input size of the layer</param>
		/// <param name="filter_window_size">2d window size of a single filter</param>
		/// <param name="filters_count">Number of filters (that defines number of output channels)</param>
		/// <param name="func_id">Identifier of an activation function to use</param>
		/// <param name="paddings">Zero paddings to be applied to the input tensor when running convolution with the filters</param>
		/// <param name="strides">Stride used in the convolution</param>
		void initialize(const Index3d& in_size, const Index2d& filter_window_size,
			const std::size_t& filters_count, const ActivationFunctionId func_id, const Index3d& paddings, const Index3d& strides);
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
		CLayer() = default;

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
			const std::size_t& filters_count, const ActivationFunctionId func_id, const Index3d& paddings = { 0 }, const Index3d& strides = {1});

		/// <summary>
		/// Constructor to instantiate layer from the given string of certain format
		/// </summary>
		CLayer(const std::string& str);

		/// <summary>
		/// See description in the base class
		/// </summary>
		typename D::tensor_t act(const typename D::tensor_t& input, typename ALayer<D>::AuxLearningData* const aux_learning_data_ptr = nullptr) const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual std::tuple<typename D::tensor_t, typename ALayer<D>::LayerGradient> backpropagate(const typename D::tensor_t& deltas, const typename ALayer<D>::AuxLearningData& aux_learning_data,
			const bool evaluate_input_gradient = true) const override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual void update(const std::tuple<std::vector<typename D::tensor_t>, typename D::tensor_t>& weights_and_biases_increment, const Real& reg_factor) override;

		/// <summary>
		/// See description in the base class
		/// </summary>
		virtual void log(const std::filesystem::path& directory) const;

		/// <summary>
		/// See the summary to the corresponding method in the base class
		/// </summary>
		virtual std::string to_string() const override;

		/// <summary>
		/// Returns "true" if the current instance of the layer has the same set of hyper-parameters as the given one
		/// </summary>
		bool equal_hyperparams(const ALayer<D>& layer) const override;

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
		CLayer<CpuDC> to_host() const;

		/// <summary>
		/// Converts the given instance to the one working within the "gpu data context" (CUDA "device" memory)
		/// </summary>
		CLayer<GpuDC> to_device() const;
	};

}
