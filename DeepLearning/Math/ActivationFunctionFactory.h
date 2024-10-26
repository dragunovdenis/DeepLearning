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
#include "../defs.h"
#include "ActivationFunctionId.h"
#include "../CudaBridge.h"
#include "Functions.h"

namespace DeepLearning::ActivationFunctionFactory
{
	/// <summary>
	/// A factory method: the only "legal" way to instantiate an activation function via its identifier
	/// </summary>
	template <class F>
	CUDA_CALLABLE F make(const ActivationFunctionId id)
	{
		switch (id)
		{
		case ActivationFunctionId::SIGMOID: return [](const auto& x) { return Func::sigmoid(x); };
		case ActivationFunctionId::TANH: return [](const auto& x) { return tanh(x); };
		case ActivationFunctionId::RELU: return [](const auto& x) { return  x < Real(0) ? Real(0) : x; };
		case ActivationFunctionId::LINEAR: return [](const auto& x) { return  x; };
		default: return [](const auto& x) { return decltype(x)(std::numeric_limits<Real>::signaling_NaN()); };
		}
	}
}
