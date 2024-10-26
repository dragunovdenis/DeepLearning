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

#include "ActivationFunctionId.h"
#include "../Utilities.h"

namespace DeepLearning
{
	std::string to_string(const ActivationFunctionId& activation_type_id)
	{
		switch (activation_type_id)
		{
		case ActivationFunctionId::SIGMOID: return "SIGMOID";
		case ActivationFunctionId::TANH: return "TANH";
		case ActivationFunctionId::RELU: return "RELU";
		case ActivationFunctionId::SOFTMAX: return "SOFTMAX";
		case ActivationFunctionId::LINEAR: return "LINEAR";
		case ActivationFunctionId::UNKNOWN: return "UNKNOWN";
		default:
			throw std::exception("Unknown activation ID");
		}
	}

	ActivationFunctionId parse_activation_type(const std::string& str)
	{
		const auto str_normalized = Utils::normalize_string(str);

		for (auto id = static_cast<unsigned int>(ActivationFunctionId::SIGMOID);
			id <= static_cast<unsigned int>(ActivationFunctionId::LINEAR); id++)
		{
			if (to_string(static_cast<ActivationFunctionId>(id)) == str_normalized)
				return static_cast<ActivationFunctionId>(id);
		}

		return ActivationFunctionId::UNKNOWN;
	}
}