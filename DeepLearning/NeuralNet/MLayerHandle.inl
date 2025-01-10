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
#include "RMLayer.h"
#include "MLayerHandle.h"
#include <exception>

namespace DeepLearning
{
	template <class D>
	MLayerHandle<D>::MLayerHandle(const MLayerTypeId id, std::unique_ptr<AMLayer<D>>&& layer_ptr) :
	_layer_id(id), _layer_ptr(std::move(layer_ptr))
	{}

	template <class D>
	MLayerHandle<D>::MLayerHandle(const MLayerHandle& handle) : _layer_id(handle._layer_id)
	{
		if (_layer_id == RMLayer<D>::ID())
		{
			_layer_ptr = std::make_unique<RMLayer<D>>(dynamic_cast<const RMLayer<D>&>(handle.layer()));
		}
		else throw std::exception("Unsupported layer type ID");
	}

	template <class D>
	void MLayerHandle<D>::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		//Read identifier only
		msgpack::type::make_define_array(_layer_id).msgpack_unpack(msgpack_o);

		if (_layer_id == RMLayer<D>::ID())
		{
			auto proxy = RMLayer<D>();
			//Read once again, but this time we read the instance of the layer as well
			msgpack::type::make_define_array(_layer_id, proxy).msgpack_unpack(msgpack_o);
			_layer_ptr = std::make_unique<decltype(proxy)>(std::move(proxy));
			return;
		}

		throw std::exception("Not implemented");
	}

	template <class D>
	template <typename Packer>
	void MLayerHandle<D>::msgpack_pack(Packer& msgpack_pk) const
	{
		if (_layer_id == RMLayer<D>::ID())
		{
			const auto& layer_ref_casted = dynamic_cast<const RMLayer<D>&>(layer());
			msgpack::type::make_define_array(_layer_id, layer_ref_casted).msgpack_pack(msgpack_pk);
			return;
		}

		throw std::exception("Not implemented");
	}

	template <class D>
	template <template <typename> class L, class... Types>
	MLayerHandle<D> MLayerHandle<D>::make(Types&&... args)
	{
		return MLayerHandle(L<D>::ID(), std::make_unique<L<D>>(std::forward<Types>(args)...));
	}

	template <class D>
	AMLayer<D>& MLayerHandle<D>::layer()
	{
		return *_layer_ptr;
	}

	template <class D>
	const AMLayer<D>& MLayerHandle<D>::layer() const
	{
		return *_layer_ptr;
	}
	template<class D>
	bool MLayerHandle<D>::operator==(const MLayerHandle& handle) const
	{
		return _layer_id == handle._layer_id && layer().equal(handle.layer());
	}
	template<class D>
	bool MLayerHandle<D>::operator!=(const MLayerHandle& handle) const
	{
		return !(*this == handle);
	}
}
