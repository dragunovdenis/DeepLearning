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
		default: throw std::exception("Unexpected activation function ID.");
			break;
		}
	}


	DenseVector ActivationFuncion::operator ()(const DenseVector& input) const
	{
		DenseVector result(input.dim());
		std::transform(input.begin(), input.end(), result.begin(),
			[&](const auto& x) { return  _func->operator()(x); });
		return result;
	}

	std::tuple<DenseVector, DenseVector> ActivationFuncion::func_and_deriv(const DenseVector& input) const
	{
		DenseVector func(input.dim());
		DenseVector deriv(input.dim());

		for (std::size_t item_id = 0; item_id < input.dim(); item_id++)
		{
			const auto [value, derivative] = _func->calc_funcion_and_derivative(input(item_id));
			func(item_id) = value;
			deriv(item_id) = derivative;
		}

		return std::make_tuple(func, deriv);
	}
}