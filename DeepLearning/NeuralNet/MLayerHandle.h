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
#include <memory>
#include "AMLayer.h"
#include "MLayerTypeId.h"

namespace DeepLearning
{
	/// <summary>
	/// A proxy type to handle serialization of different
	/// implementations of AMLayer in a unified way.
	/// </summary>
	template <class D>
	class MLayerHandle
	{
		/// <summary>
		/// Layer identifier
		/// </summary>
		MLayerTypeId _layer_id = MLayerTypeId::UNKNOWN;

		/// <summary>
		/// Layer pointer
		/// </summary>
		std::unique_ptr<AMLayer<D>> _layer_ptr = nullptr;

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="id">Layer identifier</param>
		/// <param name="layer_ptr">Layer smart pointer</param>
		MLayerHandle(const MLayerTypeId id, std::unique_ptr<AMLayer<D>>&& layer_ptr);

	public:

		/// <summary>
		/// Default constructor
		/// </summary>
		MLayerHandle() = default;

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const;

		/// <summary>
		/// Copy constructor
		/// </summary>
		MLayerHandle(const MLayerHandle& handle);

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

		/// <summary>
		/// Factory method
		/// </summary>
		template <template <typename> class L, class... Types>
		static MLayerHandle make(Types&&... args);

		/// <summary>
		/// Reference to the layer
		/// </summary>
		AMLayer<D>& layer();

		/// <summary>
		/// Reference to the layer (constant version)
		/// </summary>
		const AMLayer<D>& layer() const;

		/// <summary>
		/// Returns "true" if the current instance is equal to the given one.
		/// </summary>
		bool operator ==(const MLayerHandle& handle) const;

		/// <summary>
		/// Returns "true" if the current instance is not equal to the given one;
		/// </summary>
		bool operator!=(const MLayerHandle& handle) const;
	};
}

#include "MLayerHandle.inl"