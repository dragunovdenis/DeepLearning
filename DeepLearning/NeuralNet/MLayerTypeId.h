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
#include <string>

namespace DeepLearning
{
	/// <summary>
	/// Represents different type of multi-layers.
	/// </summary>
	enum class MLayerTypeId : int
	{
		UNKNOWN = 0,
		RECURRENT = 1,
		UNI_FULLY_CONNECTED = 2,
	};

	/// <summary>
	/// Returns string representation of the given multi-layer type ID.
	/// </summary>
	std::string to_string(const MLayerTypeId& type_id);

	/// <summary>
	/// Retrieves multi-layer type ID from the given string representation.
	/// </summary>
	MLayerTypeId parse_multi_layer_type(const std::string& str);
}

MSGPACK_ADD_ENUM(DeepLearning::MLayerTypeId)