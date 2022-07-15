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
#include "CudaVector.cuh"
#include "CudaMatrix.cuh"
#include "CudaTensor.cuh"

namespace DeepLearning
{
	namespace ActivationFunctionHelper
	{
		void evaluate_in_place(BasicCollection& collection, const ActivationFunctionId id)
		{
			const auto func = make<std::function<Real(Real)>>(id);

			std::transform(collection.begin(), collection.end(), collection.begin(),
				[&](const auto& x) { return  func(x); });
		}

		void evaluate_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv, const ActivationFunctionId id)
		{
			const auto func_ = make<std::function<dual<Real>(dual<Real>)>>(id);
			for (std::size_t item_id = 0; item_id < collection_func.size(); item_id++)
			{
				const auto res = func_({ collection_func.begin()[item_id], Real(1) });
				collection_func.begin()[item_id] = res.Real();
				collection_deriv.begin()[item_id] = res.Dual()[0];
			}
		}

		void normalize_and_evaluate_exponent_in_place(BasicCollection& collection)
		{
			const auto max_element = collection.max_element();
			std::transform(collection.begin(), collection.end(), collection.begin(), [max_element](const auto& x) { return std::exp(x - max_element); });
		}

		void evaluate_softmax_input_grad(const BasicCollection& input_exp, const BasicCollection& out_grad, BasicCollection& result)
		{
			const auto one_over_denominator = Real(1) / input_exp.sum();
			const auto one_over_denominator_squared = one_over_denominator * one_over_denominator;

			std::transform(result.begin(), result.end(), out_grad.begin(), result.begin(),
				[one_over_denominator](const auto& x, const auto& y) { return x * y * one_over_denominator; });
			const auto temp_sum = result.sum() * one_over_denominator;
			std::transform(result.begin(), result.end(), input_exp.begin(), result.begin(),
				[temp_sum](const auto& x, const auto& a) { return x - a * temp_sum; });
		}
	}

	template <class T>
	ActivationFunction<T>::ActivationFunction(const ActivationFunctionId id) : _id{id} {}

	template <class T>
	T ActivationFunction<T>::operator ()(const T& input) const
	{
		auto result = input;
		ActivationFunctionHelper::evaluate_in_place(result, _id);
		return result;
	}

	template <class T>
	std::tuple<T, T> ActivationFunction<T>::func_and_aux(const T& input) const
	{
		auto func = input;
		T deriv(input.size_3d(), false /*assign zero*/);
		ActivationFunctionHelper::evaluate_in_place(func, deriv, _id);
		return std::make_tuple(std::move(func), std::move(deriv));
	}

	template <class T>
	T ActivationFunction<T>::calc_input_gradient(const typename T::Base& out_grad, const T& aux_data) const
	{
		return out_grad.hadamard_prod(aux_data);
	}

	template <class T>
	T SoftMaxActivationFunction<T>::operator ()(const T& input) const
	{
		auto result = input;
		ActivationFunctionHelper::normalize_and_evaluate_exponent_in_place(result);
		const auto factor = Real(1) / result.sum();
		result.mul(factor);

		return result;
	}

	template <class T>
	std::tuple<T, T> SoftMaxActivationFunction<T>::func_and_aux(const T& input) const
	{
		auto aux_data = input;
		ActivationFunctionHelper::normalize_and_evaluate_exponent_in_place(aux_data);
		const auto factor = Real(1) / aux_data.sum();
		auto result = aux_data; 
		result.mul(factor);

		return std::make_tuple(std::move(result), std::move(aux_data));
	}

	template <class T>
	T SoftMaxActivationFunction<T>::calc_input_gradient(const typename T::Base& out_grad, const T& aux_data) const
	{
		if (out_grad.size() != aux_data.size())
			throw std::exception("Inconsistent input data");

		auto result = aux_data;
		ActivationFunctionHelper::evaluate_softmax_input_grad(aux_data, out_grad, result);

		return result;
	}

	template <class T>
	ActivationWrapper<T>::ActivationWrapper(const ActivationFunctionId id)
	{
		if (id == ActivationFunctionId::SOFTMAX)
		{
			_func = std::make_unique<SoftMaxActivationFunction<T>>();
		} else
			_func =  std::make_unique<ActivationFunction<T>>(id);
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

	template class ActivationFunction<Vector>;
	template class ActivationFunction<CudaVector>;
	template class ActivationFunction<Matrix>;
	template class ActivationFunction<CudaMatrix>;
	template class ActivationFunction<Tensor>;
	template class ActivationFunction<CudaTensor>;

	template class SoftMaxActivationFunction<Vector>;
	template class SoftMaxActivationFunction<CudaVector>;
	template class SoftMaxActivationFunction<Matrix>;
	template class SoftMaxActivationFunction<CudaMatrix>;
	template class SoftMaxActivationFunction<Tensor>;
	template class SoftMaxActivationFunction<CudaTensor>;

	template class ActivationWrapper<Vector>;
	template class ActivationWrapper<CudaVector>;
	template class ActivationWrapper<Matrix>;
	template class ActivationWrapper<CudaMatrix>;
	template class ActivationWrapper<Tensor>;
	template class ActivationWrapper<CudaTensor>;
}