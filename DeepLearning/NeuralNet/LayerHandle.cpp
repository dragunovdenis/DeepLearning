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

#include "LayerHandle.h"

namespace DeepLearning
{
	LayerHandle::LayerHandle(const LayerTypeId id, std::unique_ptr<ALayer> layer_ptr) : _layer_id(id), _layer_ptr(std::move(layer_ptr))
	{}

	void LayerHandle::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		//Read identifier only
		msgpack::type::make_define_array(_layer_id).msgpack_unpack(msgpack_o);
		auto proxy = get_layer_instance(_layer_id);
		//Read once again, but this time we read the instance of the layer as well
		msgpack::type::make_define_array(_layer_id, proxy).msgpack_unpack(msgpack_o);
		_layer_ptr = std::make_unique<decltype(proxy)>(std::move(proxy));
	}

	ALayer& LayerHandle::layer()
	{
		if (_layer_ptr == nullptr)
			throw std::exception("Layer is not initialized");

		return *_layer_ptr;
	}

	const ALayer& LayerHandle::layer() const
	{
		if (_layer_ptr == nullptr)
			throw std::exception("Layer is not initialized");

		return *_layer_ptr;
	}
}