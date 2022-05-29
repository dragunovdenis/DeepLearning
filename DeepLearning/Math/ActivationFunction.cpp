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
#include "Vector.h"
#include "Matrix.h"
#include "Tensor.h"
#include <vector>
#include <algorithm>
#include <exception>

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

	template <class T>
	ActivationFuncion<T>::ActivationFuncion(const ActivationFunctionId id)
	{
		switch (id)
		{
		case ActivationFunctionId::UNKNOWN: throw std::exception("Invalid activation function ID.");
			break;
		case ActivationFunctionId::SIGMOID: _func = DiffFunc::create([](const auto& x, const auto& param) { return sigmoid(x); });
			break;
		case ActivationFunctionId::TANH: _func = DiffFunc::create([](const auto& x, const auto& param) { return tanh(x); });
			break;
		case ActivationFunctionId::RELU: _func = DiffFunc::create([](const auto& x, const auto& param) { return  x < Real(0) ? Real(0) : x; });
			break;
		default: throw std::exception("Unexpected activation function ID.");
			break;
		}
	}

	template <class T>
	T ActivationFuncion<T>::operator ()(const T& input) const
	{
		auto result = input;
		std::transform(result.begin(), result.end(), result.begin(),
			[&](const auto& x) { return  _func->operator()(x); });
		return result;
	}

	template <class T>
	std::tuple<T, T> ActivationFuncion<T>::func_and_aux(const T& input) const
	{
		auto func = input;
		auto deriv = input;

		for (std::size_t item_id = 0; item_id < input.size(); item_id++)
		{
			const auto [value, derivative] = _func->calc_funcion_and_derivative(func.begin()[item_id]);
			func.begin()[item_id] = value;
			deriv.begin()[item_id] = derivative;
		}

		return std::make_tuple(func, deriv);
	}

	template <class T>
	T ActivationFuncion<T>::calc_input_gradient(const BasicCollection& out_grad, const T& aux_data) const
	{
		return out_grad.hadamard_prod(aux_data);
	}

	template <class T>
	T SoftMaxActivationFuncion<T>::calc_aux_data(const T& input) const
	{
		auto result = input;
		const auto max_element = result.max_element();

		std::transform(result.begin(), result.end(), result.begin(), [max_element](const auto& x) { return std::exp(x - max_element); });

		return result;
	}

	template <class T>
	T SoftMaxActivationFuncion<T>::operator ()(const T& input) const
	{
		auto result = calc_aux_data(input);
		const auto factor = Real(1) / result.sum();
		result.mul(factor);

		return result;
	}

	template <class T>
	std::tuple<T, T> SoftMaxActivationFuncion<T>::func_and_aux(const T& input) const
	{
		const auto aux_data = calc_aux_data(input);
		const auto factor = Real(1) / aux_data.sum();
		auto result = aux_data; 
		result.mul(factor);

		return std::make_tuple(result, aux_data);
	}

	template <class T>
	T SoftMaxActivationFuncion<T>::calc_input_gradient(const BasicCollection& out_grad, const T& aux_data) const
	{
		if (out_grad.size() != aux_data.size())
			throw std::exception("Inconsistent input data");

		const auto one_over_denominator = Real(1) / aux_data.sum();
		const auto one_over_denominator_squared = one_over_denominator * one_over_denominator;

		auto result = aux_data;
		std::transform(result.begin(), result.end(), out_grad.begin(), result.begin(), [one_over_denominator](const auto& x, const auto& y) { return x * y * one_over_denominator; });
		const auto temp_sum = result.sum() * one_over_denominator;
		std::transform(result.begin(), result.end(), aux_data.begin(), result.begin(), [temp_sum](const auto& x, const auto& a) { return x - a * temp_sum; });

		return result;
	}

	template <class T>
	ActivationWrapper<T>::ActivationWrapper(const ActivationFunctionId id)
	{
		if (id == ActivationFunctionId::SOFTMAX)
		{
			_func = std::make_unique<SoftMaxActivationFuncion<T>>();
		} else
			_func =  std::make_unique<ActivationFuncion<T>>(id);
	}

	template <class T>
	const AFunction<T>& ActivationWrapper<T>::operator()() const
	{
		return *_func;
	}

	std::string to_string(const ActivationFunctionId& activation_type_id)
	{
		switch (activation_type_id)
		{
		case ActivationFunctionId::SIGMOID: return "SIGMOID";
		case ActivationFunctionId::TANH: return "TANH";
		case ActivationFunctionId::RELU: return "RELU";
		case ActivationFunctionId::SOFTMAX: return "SOFTMAX";
		default:
			return "UNKNOWN";
		}
	}

	ActivationFunctionId parse_activation_type(const std::string& str)
	{
		const auto str_normalized = Utils::normalize_string(str);

		for (unsigned int id = (unsigned int)ActivationFunctionId::SIGMOID; id <= (unsigned int)ActivationFunctionId::SOFTMAX; id++)
		{
			if (to_string((ActivationFunctionId)id) == str_normalized)
				return (ActivationFunctionId)id;
		}

		return ActivationFunctionId::UNKNOWN;
	}

	template class ActivationFuncion<Vector>;
	template class ActivationFuncion<Matrix>;
	template class ActivationFuncion<Tensor>;

	template class SoftMaxActivationFuncion<Vector>;
	template class SoftMaxActivationFuncion<Matrix>;
	template class SoftMaxActivationFuncion<Tensor>;

	template class ActivationWrapper<Vector>;
	template class ActivationWrapper<Matrix>;
	template class ActivationWrapper<Tensor>;
}