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
#include <exception>

namespace DeepLearning
{
	template <class T>
	ActivationFunction<T>::ActivationFunction(const ActivationFunctionId id) : _id{id} {}

	template <class T>
	void ActivationFunction<T>::func_in_place(T& in_out) const
	{
		T::ActivationHelper::evaluate_in_place(in_out, _id);
	}

	template <class T>
	void ActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		aux.resize(in_out.size_3d());
		T::ActivationHelper::evaluate_in_place(in_out, aux, _id);
	}

	template <class T>
	void SoftMaxActivationFunction<T>::func_in_place(T& in_out) const
	{
		T::ActivationHelper::normalize_and_evaluate_exponent_in_place(in_out);
		const auto factor = static_cast<Real>(1) / in_out.sum();
		in_out.mul(factor);
	}

	template <class T>
	void SoftMaxActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		T::ActivationHelper::normalize_and_evaluate_exponent_in_place(in_out);
		const auto factor = static_cast<Real>(1) / in_out.sum();
		aux = in_out;
		in_out.mul(factor);
	}

	template <class T>
	void SoftMaxActivationFunction<T>::calc_in_grad(const typename T::Base& out_grad, const T& aux_data, T& result) const
	{
		if (out_grad.size() != aux_data.size())
			throw std::exception("Inconsistent input data");

		T::Base::ActivationHelper::evaluate_softmax_input_grad(aux_data, out_grad, result);
	}

	template<class T>
	void SoftMaxActivationFunction<T>::add_in_grad(const typename T::Base& out_grad, const T& aux_data, T& out_res) const
	{
		if (out_grad.size() != aux_data.size())
			throw std::exception("Inconsistent input data");

		thread_local T temp{};
		temp.resize(out_res.size_3d());
		T::Base::ActivationHelper::evaluate_softmax_input_grad(aux_data, out_grad, temp);
		out_res += temp;
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

	template<class T>
	void AFunction<T>::add_in_grad(const typename T::Base& out_grad, const T& aux_data, T& out_res) const
	{
		out_res.hadamard_prod_add(aux_data, out_grad);
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
		T::ActivationHelper::relu_in_place(in_out);
	}

	template <class T>
	void ReLuActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		aux.resize(in_out.size_3d());
		T::ActivationHelper::relu_in_place(in_out, aux);
	}

	template <class T>
	void SigmoidActivationFunction<T>::func_in_place(T& in_out) const
	{
		T::ActivationHelper::sigmoid_in_place(in_out);
	}

	template <class T>
	void SigmoidActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		aux.resize(in_out.size_3d());
		T::ActivationHelper::sigmoid_in_place(in_out, aux);
	}

	template <class T>
	void TanhActivationFunction<T>::func_in_place(T& in_out) const
	{
		T::ActivationHelper::tanh_in_place(in_out);
	}

	template <class T>
	void TanhActivationFunction<T>::func_and_aux_in_place(T& in_out, T& aux) const
	{
		aux.resize(in_out.size_3d());
		T::ActivationHelper::tanh_in_place(in_out, aux);
	}
}