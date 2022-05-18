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

#include "CostFunction.h"
#include "Vector.h"
#include "Matrix.h"
#include "Tensor.h"
#include <exception>
#include <numeric>
#include "../Utilities.h"

namespace DeepLearning
{
	CostFunction::CostFunction(const CostFunctionId id)
	{
		switch (id)
		{
		case CostFunctionId::UNKNOWN: throw std::exception("Invalid activation function ID.");
			break;
		case CostFunctionId::SQUARED_ERROR: _func = DiffFunc::create(
			[](const auto& x, const auto& ref) 
			{ 
				const auto diff = x - ref;
				return Real(0.5)* diff * diff;
			});
			break;
		case CostFunctionId::CROSS_ENTROPY:_func = DiffFunc::create(
			[](const auto& x, const auto& ref)
			{
				if (ref <= Real(0))
					return Utils::nan_to_num(-log(Real(1) - x));

				if (ref >= Real(1))
					return Utils::nan_to_num(-log(x));

				return Utils::nan_to_num(-(ref*log(x) + (Real(1) - ref)*log(Real(1) - x)));
			});
			break;
		default: throw std::exception("Unexpected cost function ID.");
			break;
		}
	}

	template <class T>
	Real CostFunction::operator ()(const T& output, const T& reference) const
	{
		if (output.size() != reference.size())
			throw std::exception("Incompatible input");

		const auto result = std::transform_reduce(output.begin(), output.end(), reference.begin(), Real(0), std::plus<Real>(),
			[&](const auto& x, const auto& ref) { return  _func->operator()(x, ref); });
		return result;
	}

	template <class T>
	std::tuple<Real, T> CostFunction::func_and_deriv(const T& output, const T& reference) const
	{
		if (output.size() != reference.size())
			throw std::exception("Incompatible input");

		auto deriv = output;
		auto func_val = Real(0);

		for (auto item_id = 0ull; item_id < output.size(); item_id++)
		{
			const auto [value, derivative] = _func->calc_funcion_and_derivative(deriv.begin()[item_id], reference.begin()[item_id]);
			func_val += value;
			deriv.begin()[item_id] = derivative;
		}

		return std::make_tuple(func_val, deriv);
	}

	template Real CostFunction::operator ()(const Vector& output, const Vector& reference) const;
	template Real CostFunction::operator ()(const Matrix& output, const Matrix& reference) const;
	template Real CostFunction::operator ()(const Tensor& output, const Tensor& reference) const;

	template std::tuple<Real, Vector> CostFunction::func_and_deriv(const Vector& output, const Vector& reference) const;
	template std::tuple<Real, Matrix> CostFunction::func_and_deriv(const Matrix& output, const Matrix& reference) const;
	template std::tuple<Real, Tensor> CostFunction::func_and_deriv(const Tensor& output, const Tensor& reference) const;

}