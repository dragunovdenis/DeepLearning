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
#include <msgpack.hpp>
#include <memory>
#include "ALayer.h"
#include "NeuralLayer.h"
#include <exception>

namespace DeepLearning
{
	/// <summary>
	/// Enumerable describes different types of neural layers
	/// </summary>
	enum class LayerTypeId : unsigned int {
		UNKNOWN = 0,
	    FULL = 1, //a fully connected neural layer
		CONVOLUTION = 2,//a convolution neural layer
		PULL = 3,//a pulling neural layer 
	};

	/// <summary>
	/// A proxy type to handle serialization of layers of different types
	/// </summary>
	class LayerHandle
	{
		/// <summary>
		/// Layer identifier
		/// </summary>
		LayerTypeId _layer_id = LayerTypeId::UNKNOWN;

		/// <summary>
		/// Layer pointer
		/// </summary>
		std::unique_ptr<ALayer> _layer_ptr = nullptr;

		/// <summary>
		/// Returns default instance of a layer container associated with the given identifier
		/// </summary>
		static auto get_layer_instance(const LayerTypeId id)
		{
			switch (id)
			{
			case DeepLearning::LayerTypeId::FULL:
				return NeuralLayer();
			case DeepLearning::LayerTypeId::CONVOLUTION:
				throw std::exception("Not implemented option");
				break;
			case DeepLearning::LayerTypeId::PULL:
				throw std::exception("Not implemented option");
				break;
			default:
				throw std::exception("Unsupported layer identifier");
				break;
			}
		}

		/// <summary>
		/// Factory method
		/// </summary>
		template <class... Types>
		static std::unique_ptr<ALayer> make_layer_ptr(const LayerTypeId id, Types&&... args)
		{
			return std::make_unique<decltype(get_layer_instance(id))>(std::forward<Types>(args)...);
		}

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="id">Layer identifier</param>
		/// <param name="layer_ptr">Layer smart pointer</param>
		LayerHandle(const LayerTypeId id, std::unique_ptr<ALayer> layer_ptr);

	public:

		/// <summary>
		/// Default constructor
		/// </summary>
		LayerHandle() {}

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const
		{
			const auto& layer_ref_casted = static_cast<const decltype(get_layer_instance(_layer_id))&>(layer());
			msgpack::type::make_define_array(_layer_id, layer_ref_casted).msgpack_pack(msgpack_pk);
		}

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

		/// <summary>
		/// Factory method
		/// </summary>
		template <class... Types>
		static LayerHandle make(const LayerTypeId id, Types&&... args)
		{
			return LayerHandle(id, make_layer_ptr(id, std::forward<Types>(args)...));
		}

		/// <summary>
		/// Reference to the layer
		/// </summary>
		ALayer& layer();

		/// <summary>
		/// Reference to the layer (constant version)
		/// </summary>
		const ALayer& layer() const;
	};
}

MSGPACK_ADD_ENUM(DeepLearning::LayerTypeId)

