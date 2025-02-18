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

#include "LayerTypeId.h"
#include "../Utilities.h"

namespace DeepLearning
{
	std::string to_string(const LayerTypeId& layer_type_id)
	{
		switch (layer_type_id)
		{
		case LayerTypeId::CONVOLUTION: return "CONV";
		case LayerTypeId::FULL: return "FULL";
		case LayerTypeId::PULL: return "PULL";
		default:
			return "UNKNOWN";
		}
	}

	LayerTypeId parse_layer_type(const std::string& str)
	{
		const auto str_normalized = Utils::normalize_string(str);

		for (unsigned int id = static_cast<unsigned int>(LayerTypeId::FULL); id <= static_cast<unsigned int>(LayerTypeId::PULL); id++)
		{
			if (to_string(static_cast<LayerTypeId>(id)) == str_normalized)
				return static_cast<LayerTypeId>(id);
		}

		return LayerTypeId::UNKNOWN;
	}
}