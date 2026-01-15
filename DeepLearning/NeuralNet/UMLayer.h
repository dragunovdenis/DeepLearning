//Copyright (c) 2026 Denys Dragunov, dragunovdenis@gmail.com
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
#include "ALayer.h"
#include "NLayer.h"

namespace DeepLearning
{
	/// <summary>
	/// A general implementation of a uni-modal multi layer, i.e.,
	/// a layer that applies the same core layer to each depth slice
	/// of the input data.
	/// </summary>
	template <class D, template<class> class L>
	class UMLayer : public AMLayer<D>
	{
		L<D> _core;

		static_assert(std::is_base_of_v<ALayer<D>, L<D>>,
			"L<D> must be derived from ALayer<D>");

		static constexpr int MSG_PACK_VER = 1;

		long long _depth{ -1 };

		/// <summary>
		/// Returns reference to the core layer.
		/// </summary>
		ALayer<D>& core_layer();

		/// <summary>
		/// Returns constant reference to the core layer.
		/// </summary>
		const ALayer<D>& core_layer() const;

	public:
		/// <summary>
		/// Constructor.
		/// </summary>
		template <typename... Args>
		UMLayer(const long long, Args... core_layer_args);

		/// <summary>
		/// Default constructor.
		/// </summary>
		UMLayer() = default;

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		Index4d in_size() const override;

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		Index4d out_size() const override;

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		MLayerGradient<D> allocate_gradient_container(const bool fill_zero = false) const override;

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		MLayerData<D> allocate_trace_data() const override;

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		void act(const IMLayerExchangeData<typename D::tensor_t>& input,
			IMLayerExchangeData<typename D::tensor_t>& output, IMLayerTraceData<D>* const trace_data) const override;

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		void backpropagate(const IMLayerExchangeData<typename D::tensor_t>& out_grad,
			const IMLayerExchangeData<typename D::tensor_t>& output,
			const IMLayerExchangeData<LayerData<D>>& processing_data,
			IMLayerExchangeData<typename D::tensor_t>& out_input_grad,
			MLayerGradient<D>& out_layer_grad, const bool evaluate_input_gradient = true) const override;

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		bool equal_hyperparams(const AMLayer<D>& layer) const override;

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		bool equal(const AMLayer<D>& layer) const override;

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		void update(const MLayerGradient<D>& increment, const Real learning_rate, const Real reg_factor) override;

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const;
	};


	/// <summary>
	/// Uni-modal fully connected multi-layer.
	/// </summary>
	template <class D = CpuDC>
	class UNMLayer : public UMLayer<D, NLayer>
	{
	public:

		/// <summary>
		/// Default constructor.
		/// </summary>
		UNMLayer() = default;

		/// <summary>
		/// Constructor.
		/// </summary>
		/// <param name="in_size">Dimensionality of the layer's input.</param>
		/// <param name="out_size">Dimensionality of the layer's output.</param>
		/// <param name="func_id">Identifier of the activation function to be used by the layer.</param>
		UNMLayer(const Index4d& in_size, const Index4d& out_size, ActivationFunctionId func_id): UMLayer<D, NLayer>(in_size.w,
				in_size.xyz.coord_prod(), out_size.xyz.coord_prod(), func_id, static_cast<Real>(-1),
				static_cast<Real>(1), true, static_cast<Real>(1))
		{
			if (in_size.w != out_size.w)
				throw std::exception("Input depth must coincide with the output depth.");
		}

		/// <summary>
		/// See the summary of the base class method.
		/// </summary>
		MLayerTypeId get_type_id() const override;

		/// <summary>
		/// Returns ID of the class.
		/// </summary>
		static constexpr  MLayerTypeId ID() { return MLayerTypeId::UNI_FULLY_CONNECTED; };
	};
}

#include "UMLayer.inl"
