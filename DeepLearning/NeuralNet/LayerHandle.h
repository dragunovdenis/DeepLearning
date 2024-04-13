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
#include <memory>
#include "ALayer.h"
#include "NLayer.h"
#include "CLayer.h"
#include "PLayer.h"
#include <exception>
#include "LayerTypeId.h"

namespace DeepLearning
{
	/// <summary>
	/// A proxy type to handle serialization of layers of different types
	/// </summary>
	template <class D>
	class LayerHandle
	{
		/// <summary>
		/// Layer identifier
		/// </summary>
		LayerTypeId _layer_id = LayerTypeId::UNKNOWN;

		/// <summary>
		/// Layer pointer
		/// </summary>
		std::unique_ptr<ALayer<D>> _layer_ptr = nullptr;

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="id">Layer identifier</param>
		/// <param name="layer_ptr">Layer smart pointer</param>
		LayerHandle(const LayerTypeId id, std::unique_ptr<ALayer<D>>&& layer_ptr);

	public:

		/// <summary>
		/// Default constructor
		/// </summary>
		LayerHandle() = default;

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const
		{
			if (_layer_id == NLayer<D>::ID())
			{
				const auto& layer_ref_casted = dynamic_cast<const NLayer<D>&>(layer());
				msgpack::type::make_define_array(_layer_id, layer_ref_casted).msgpack_pack(msgpack_pk);
				return;
			}

			if (_layer_id == CLayer<D>::ID())
			{
				const auto& layer_ref_casted = dynamic_cast<const CLayer<D>&>(layer());
				msgpack::type::make_define_array(_layer_id, layer_ref_casted).msgpack_pack(msgpack_pk);
				return;
			}

			if (_layer_id == PLayer<D>::ID())
			{
				const auto& layer_ref_casted = dynamic_cast<const PLayer<D>&>(layer());
				msgpack::type::make_define_array(_layer_id, layer_ref_casted).msgpack_pack(msgpack_pk);
				return;
			}

			throw std::exception("Not implemented");
		}

		/// <summary>
		/// Copy constructor
		/// </summary>
		LayerHandle(const LayerHandle<D>& handle);

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

		/// <summary>
		/// Factory method
		/// </summary>
		template <class L, class... Types>
		static LayerHandle make(Types&&... args)
		{
			return LayerHandle(L::ID(), std::make_unique<L>(std::forward<Types>(args)...));
		}

		/// <summary>
		/// Reference to the layer
		/// </summary>
		ALayer<D>& layer();

		/// <summary>
		/// Reference to the layer (constant version)
		/// </summary>
		const ALayer<D>& layer() const;
	};
}

