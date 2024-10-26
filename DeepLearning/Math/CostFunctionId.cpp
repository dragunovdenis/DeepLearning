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

#include "CostFunctionId.h"
#include "../Utilities.h"

namespace DeepLearning
{
	std::string to_string(const CostFunctionId& cost_type_id)
	{
		switch (cost_type_id)
		{
		case CostFunctionId::SQUARED_ERROR: return "SQUARED_ERROR";
		case CostFunctionId::CROSS_ENTROPY: return "CROSS_ENTROPY";
		case CostFunctionId::LINEAR: return "LINEAR";
		default:
			return "UNKNOWN";
		}
	}

	CostFunctionId parse_cost_type(const std::string& str)
	{
		const auto str_normalized = Utils::normalize_string(str);

		for (auto id = static_cast<unsigned int>(CostFunctionId::SQUARED_ERROR);
			id <= static_cast<unsigned int>(CostFunctionId::LINEAR); id++)
		{
			if (to_string(static_cast<CostFunctionId>(id)) == str_normalized)
				return static_cast<CostFunctionId>(id);
		}

		return CostFunctionId::UNKNOWN;
	}
}
