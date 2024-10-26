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

#include "ActivationFunctionHelper.h"
#include "ActivationFunctionFactory.h"
#include "BasicCollection.h"
#include "Dual.h"

namespace DeepLearning
{
	void ActivationFunctionHelper::evaluate_in_place(BasicCollection& collection, const ActivationFunctionId id)
	{
		const auto func = ActivationFunctionFactory::make<std::function<Real(Real)>>(id);
		std::ranges::transform(collection, collection.begin(), [&](const auto& x) { return  func(x); });
	}

	void ActivationFunctionHelper::evaluate_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv, const ActivationFunctionId id)
	{
		const auto func_ = ActivationFunctionFactory::make<std::function<dual<Real>(dual<Real>)>>(id);
		for (std::size_t item_id = 0; item_id < collection_func.size(); item_id++)
		{
			const auto res = func_({ collection_func.begin()[item_id], static_cast<Real>(1) });
			collection_func.begin()[item_id] = res.Real();
			collection_deriv.begin()[item_id] = res.Dual()[0];
		}
	}

	void ActivationFunctionHelper::normalize_and_evaluate_exponent_in_place(BasicCollection& collection)
	{
		const auto max_element = collection.max_element();
		std::ranges::transform(collection, collection.begin(), [max_element](const auto& x) { return std::exp(x - max_element); });
	}

	void ActivationFunctionHelper::evaluate_softmax_input_grad(const BasicCollection& input_exp, const BasicCollection& out_grad, BasicCollection& result)
	{
		const auto one_over_denominator = static_cast<Real>(1) / input_exp.sum();
		std::transform(input_exp.begin(), input_exp.end(), out_grad.begin(), result.begin(),
			[one_over_denominator](const auto& x, const auto& y) { return x * y * one_over_denominator; });
		const auto temp_sum = result.sum() * one_over_denominator;
		std::transform(result.begin(), result.end(), input_exp.begin(), result.begin(),
			[temp_sum](const auto& x, const auto& a) { return x - a * temp_sum; });
	}

	void ActivationFunctionHelper::relu_in_place(BasicCollection& collection)
	{
		std::ranges::transform(collection, collection.begin(), [](const auto& x) { return  x > 0 ? x : 0; });
	}

	void ActivationFunctionHelper::relu_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv)
	{
		const auto func_ptr = collection_func.begin();
		const auto deriv_ptr = collection_deriv.begin();

		for (std::size_t item_id = 0; item_id < collection_func.size(); ++item_id)
		{
			const auto x = func_ptr[item_id];
			func_ptr[item_id] = x > 0 ? x : 0;
			deriv_ptr[item_id] = static_cast<Real>(x > 0 ? 1 : 0);
		}
	}

	void ActivationFunctionHelper::sigmoid_in_place(BasicCollection& collection)
	{
		std::ranges::transform(collection, collection.begin(),
			[](const auto& x) { return  1 / (1 + std::exp(-x)); });
	}

	void ActivationFunctionHelper::sigmoid_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv)
	{
		const auto func_ptr = collection_func.begin();
		const auto deriv_ptr = collection_deriv.begin();

		for (std::size_t item_id = 0; item_id < collection_func.size(); ++item_id)
		{
			const auto value = 1 / (1 + std::exp(-func_ptr[item_id]));
			func_ptr[item_id] = value;
			deriv_ptr[item_id] = value * (1 - value);
		}
	}

	void ActivationFunctionHelper::tanh_in_place(BasicCollection& collection)
	{
		std::ranges::transform(collection, collection.begin(),
			[](const auto& x) { return  std::tanh(x); });
	}

	void ActivationFunctionHelper::tanh_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv)
	{
		const auto func_ptr = collection_func.begin();
		const auto deriv_ptr = collection_deriv.begin();

		for (std::size_t item_id = 0; item_id < collection_func.size(); ++item_id)
		{
			const auto value = std::tanh(func_ptr[item_id]);
			func_ptr[item_id] = value;
			deriv_ptr[item_id] = 1 - value * value;
		}
	}
}
