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

#pragma once
#include <exception>

namespace DeepLearning
{
	template <class T>
	CostFunction<T>::CostFunction(const CostFunctionId id) : _id(id)
	{}

	template <class T>
	Real CostFunction<T>::operator ()(const T& output, const T& reference) const
	{
		if (output.size() != reference.size())
			throw std::exception("Incompatible input");

		return T::CostHelper::evaluate_cost(output, reference, _id);
	}

	template <class T>
	std::tuple<Real, T> CostFunction<T>::func_and_deriv(const T& output, const T& reference) const
	{
		if (output.size() != reference.size())
			throw std::exception("Incompatible input");

		auto deriv = output;
		const auto func_val = T::CostHelper::evaluate_cost_and_gradient(deriv, reference, _id);
		return std::make_tuple(func_val, std::move(deriv));
	}

	template <class T>
	void CostFunction<T>::deriv_in_place(T& output_deriv, const T& reference) const
	{
		T::CostHelper::evaluate_gradient(output_deriv, reference, _id);
	}

	template <class T>
	T CostFunction<T>::deriv(const T& output, const T& reference) const
	{
		auto deriv = output;
		deriv_in_place(deriv, reference);
		return deriv;
	}
}
