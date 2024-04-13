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
#include <algorithm>
#include <exception>
#include "CudaVector.cuh"
#include "CudaMatrix.cuh"
#include "CudaTensor.cuh"
#include "../Utilities.h"

namespace DeepLearning
{
	namespace ActivationFunctionHelper
	{
		void evaluate_in_place(BasicCollection& collection, const ActivationFunctionId id)
		{
			const auto func = make<std::function<Real(Real)>>(id);
			std::ranges::transform(collection, collection.begin(), [&](const auto& x) { return  func(x); });
		}

		void evaluate_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv, const ActivationFunctionId id)
		{
			const auto func_ = make<std::function<dual<Real>(dual<Real>)>>(id);
			for (std::size_t item_id = 0; item_id < collection_func.size(); item_id++)
			{
				const auto res = func_({ collection_func.begin()[item_id], static_cast<Real>(1) });
				collection_func.begin()[item_id] = res.Real();
				collection_deriv.begin()[item_id] = res.Dual()[0];
			}
		}

		void normalize_and_evaluate_exponent_in_place(BasicCollection& collection)
		{
			const auto max_element = collection.max_element();
			std::ranges::transform(collection, collection.begin(), [max_element](const auto& x) { return std::exp(x - max_element); });
		}

		void evaluate_softmax_input_grad(const BasicCollection& input_exp, const BasicCollection& out_grad, BasicCollection& result)
		{
			const auto one_over_denominator = static_cast<Real>(1) / input_exp.sum();
			std::transform(input_exp.begin(), input_exp.end(), out_grad.begin(), result.begin(),
				[one_over_denominator](const auto& x, const auto& y) { return x * y * one_over_denominator; });
			const auto temp_sum = result.sum() * one_over_denominator;
			std::transform(result.begin(), result.end(), input_exp.begin(), result.begin(),
				[temp_sum](const auto& x, const auto& a) { return x - a * temp_sum; });
		}

		void relu_in_place(BasicCollection& collection)
		{
			std::ranges::transform(collection, collection.begin(), [](const auto& x) { return  x > 0 ? x : 0; });
		}

		void relu_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv)
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

		void sigmoid_in_place(BasicCollection& collection)
		{
			std::ranges::transform(collection, collection.begin(), 
				[](const auto& x) { return  1 / (1 + std::exp(-x)); });
		}

		void sigmoid_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv)
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

		void tanh_in_place(BasicCollection& collection)
		{
			std::ranges::transform(collection, collection.begin(),
				[](const auto& x) { return  std::tanh(x); });
		}

		void tanh_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv)
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

	template <class T>
	ActivationFunction<T>::ActivationFunction(const ActivationFunctionId id) : _id{id} {}

	template <class T>
	void ActivationFunction<T>::func_in_place(T& in_out) const
	{
		ActivationFunctionHelper::evaluate_in_place(in_out, _id);
	}

	template <class T>
	void ActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		aux.resize(in_out.size_3d());
		ActivationFunctionHelper::evaluate_in_place(in_out, aux, _id);
	}

	template <class T>
	void SoftMaxActivationFunction<T>::func_in_place(T& in_out) const
	{
		ActivationFunctionHelper::normalize_and_evaluate_exponent_in_place(in_out);
		const auto factor = static_cast<Real>(1) / in_out.sum();
		in_out.mul(factor);
	}

	template <class T>
	void SoftMaxActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		ActivationFunctionHelper::normalize_and_evaluate_exponent_in_place(in_out);
		const auto factor = static_cast<Real>(1) / in_out.sum();
		aux = in_out;
		in_out.mul(factor);
	}

	template <class T>
	void SoftMaxActivationFunction<T>::calc_in_grad(const typename T::Base& out_grad, const T& aux_data, T& result) const
	{
		if (out_grad.size() != aux_data.size())
			throw std::exception("Inconsistent input data");

		ActivationFunctionHelper::evaluate_softmax_input_grad(aux_data, out_grad, result);
	}

	template <class T>
	T AFunction<T>::operator()(const T& input) const
	{
		auto result = input;
		func_in_place(result);
		return result;
	}

	template <class T>
	std::tuple<T, T> AFunction<T>::func_and_aux(const T& input) const
	{
		auto func = input;
		T deriv;
		func_and_aux_in_place(func, deriv);
		return std::make_tuple(std::move(func), std::move(deriv));
	}

	template <class T>
	T AFunction<T>::get_in_grad(const typename T::Base& out_grad, const T& aux_data) const
	{
		T result(aux_data.size_3d());
		calc_in_grad(out_grad, aux_data, result);
		return result;
	}

	template <class T>
	void AFunction<T>::calc_in_grad(const typename T::Base& out_grad, const T& aux_data, T& out) const
	{
		out.hadamard_prod(aux_data, out_grad);
	}

	template <class T>
	ActivationWrapper<T>::ActivationWrapper(const ActivationFunctionId id) : _func{ construct(id) }
	{}

	template <class T>
	const AFunction<T>& ActivationWrapper<T>::operator()() const
	{
		return *_func;
	}

	template <class T>
	std::shared_ptr<AFunction<T>> ActivationWrapper<T>::construct(const ActivationFunctionId id)
	{
		switch (id)
		{
		case ActivationFunctionId::SIGMOID:
			return std::make_shared<SigmoidActivationFunction<T>>();
		case ActivationFunctionId::RELU:
			return std::make_shared<ReLuActivationFunction<T>>();
		case ActivationFunctionId::SOFTMAX:
			return std::make_shared<SoftMaxActivationFunction<T>>();
		case ActivationFunctionId::TANH:
			return std::make_shared<TanhActivationFunction<T>>();
		default:
			return std::make_shared<ActivationFunction<T>>(id);
		}
	}

	template <class T>
	void ReLuActivationFunction<T>::func_in_place(T& in_out) const
	{
		ActivationFunctionHelper::relu_in_place(in_out);
	}

	template <class T>
	void ReLuActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		aux.resize(in_out.size_3d());
		ActivationFunctionHelper::relu_in_place(in_out, aux);
	}

	template <class T>
	void SigmoidActivationFunction<T>::func_in_place(T& in_out) const
	{
		ActivationFunctionHelper::sigmoid_in_place(in_out);
	}

	template <class T>
	void SigmoidActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		aux.resize(in_out.size_3d());
		ActivationFunctionHelper::sigmoid_in_place(in_out, aux);
	}

	template <class T>
	void TanhActivationFunction<T>::func_in_place(T& in_out) const
	{
		ActivationFunctionHelper::tanh_in_place(in_out);
	}

	template <class T>
	void TanhActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		aux.resize(in_out.size_3d());
		ActivationFunctionHelper::tanh_in_place(in_out, aux);
	}

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

	namespace
	{
#define INSTANTIATE(CLASS)                \
		template class CLASS<Vector>;     \
		template class CLASS<CudaVector>; \
		template class CLASS<Matrix>;     \
		template class CLASS<CudaMatrix>; \
		template class CLASS<Tensor>;     \
		template class CLASS<CudaTensor>;
	}

	INSTANTIATE(AFunction)
	INSTANTIATE(ActivationFunction)
	INSTANTIATE(SoftMaxActivationFunction)
	INSTANTIATE(ReLuActivationFunction)
	INSTANTIATE(TanhActivationFunction)
	INSTANTIATE(SigmoidActivationFunction)
	INSTANTIATE(ActivationWrapper)
}