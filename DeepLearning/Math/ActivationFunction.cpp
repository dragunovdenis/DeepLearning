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

#include "ActivationFunction.h"
#include "Dual.h"
#include <vector>
#include <algorithm>

namespace DeepLearning
{
	/// <summary>
	/// Sigmoid function
	/// </summary>
	template <class T>
	T sigmoid(const T& arg)
	{
		return T(1) / (T(1) + exp(-arg));
	}

	ActivationFuncion::ActivationFuncion(const ActivationFunctionId id)
	{
		switch (id)
		{
		case ActivationFunctionId::UNKNOWN: throw std::exception("Invalid activation function ID.");
			break;
		case ActivationFunctionId::SIGMOID: _func = DiffFunc::create([](const auto& x, const auto& param) { return Real(1) / (Real(1) + exp(-x)); });
			break;
		case ActivationFunctionId::TANH: _func = DiffFunc::create([](const auto& x, const auto& param) { return tanh(x); });
			break;
		case ActivationFunctionId::RELU: _func = DiffFunc::create([](const auto& x, const auto& param) { return  x < Real(0) ? Real(0) : x; });
			break;
		default: throw std::exception("Unexpected activation function ID.");
			break;
		}
	}

	Vector ActivationFuncion::operator ()(const Vector& input) const
	{
		Vector result(input.dim());
		std::transform(input.begin(), input.end(), result.begin(),
			[&](const auto& x) { return  _func->operator()(x); });
		return result;
	}

	std::tuple<Vector, Vector> ActivationFuncion::func_and_deriv(const Vector& input) const
	{
		Vector func(input.dim());
		Vector deriv(input.dim());

		for (std::size_t item_id = 0; item_id < input.dim(); item_id++)
		{
			const auto [value, derivative] = _func->calc_funcion_and_derivative(input(item_id));
			func(item_id) = value;
			deriv(item_id) = derivative;
		}

		return std::make_tuple(func, deriv);
	}
}