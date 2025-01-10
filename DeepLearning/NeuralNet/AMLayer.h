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
#include <msgpack.hpp>
#include "DataContext.h"
#include "../Math/LinAlg4d.h"
#include "MLayerGradient.h"
#include "MLayerData.h"

namespace DeepLearning
{
	/// <summary>
	/// General interface for a multi-layer.
	/// </summary>
	template <class D = CpuDC>
	struct AMLayer
	{
		/// <summary>
		/// Input dimensions of the layer.
		/// </summary>
		virtual Index4d in_size() const = 0;

		/// <summary>
		/// Output dimensions of the layer.
		/// </summary>
		virtual Index4d out_size() const = 0;

		/// <summary>
		/// Returns a container to be used in back-propagation procedure./>
		/// </summary>
		virtual MLayerGradient<D> allocate_gradient_container(const bool fill_zero = false) const = 0;

		/// <summary>
		/// Calculates result of the layer evaluated on the given <param name="input"/> data.
		/// </summary>
		/// <param name="input">Input data.</param>
		/// <param name="output">Container to receive the result.</param>
		/// <param name="trace_data">Pointer to data container to store "trace data",
		/// that later can be used during the backpropagation phase. Can be null,
		/// in which case no "trace date" will be stored.</param>
		virtual void act(const IMLayerExchangeData<typename D::tensor_t>& input,
			IMLayerExchangeData<typename D::tensor_t>& output, IMLayerTraceData<D>* const trace_data) const = 0;

		/// <summary>
		/// Calculates derivatives with respect to all the parameters of the layer.
		/// </summary>
		/// <param name="out_grad">Derivatives of the cost function with respect
		/// to the output of the layer.</param>
		/// <param name="output">The output of the layer.</param>
		/// <param name="processing_data">Intermediate data prepared during the "act"
		/// phase that is needed to calculate the derivatives.</param>
		/// <param name="out_input_grad">A container that holds gradient of the layer's input upon the method's return.</param>
		/// <param name="out_layer_grad">A container will be incremented with the layer's gradient upon the method's return.</param>
		/// <param name="evaluate_input_gradient">If "false" gradient of the layers input won't be calculated.</param>
		virtual void backpropagate(const IMLayerExchangeData<typename D::tensor_t>& out_grad,
			const IMLayerExchangeData<typename D::tensor_t>& output,
			const IMLayerExchangeData<LayerData<D>>& processing_data,
			IMLayerExchangeData<typename D::tensor_t>& out_input_grad,
			MLayerGradient<D>& out_layer_grad, const bool evaluate_input_gradient = true) const = 0;

		/// <summary>
		/// Returns "true" if the current instance of the layer has the same set of hyper-parameters as the given one
		/// </summary>
		virtual bool equal_hyperparams(const AMLayer<D>& layer) const { return true; };

		/// <summary>
		/// Returns "true" if the given layer is (absolutely) equal to the current one
		/// </summary>
		virtual bool equal(const AMLayer<D>& layer) const { return true; };

		/// <summary>
		/// Updates weights of the current layer with the given <paramref name="increment"/>
		/// multiplied by the given <paramref name="learning_rate"/>
		/// </summary>
		virtual void update(const MLayerGradient<D>& increment, const Real learning_rate) = 0;

		/// <summary>
		/// To make it possible to persist data on this
		/// level without braking backward compatibility.
		/// </summary>
		MSGPACK_DEFINE();
	};
}
