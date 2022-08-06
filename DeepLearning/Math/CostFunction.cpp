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
#include "CudaVector.cuh"
#include "CudaMatrix.cuh"
#include "CudaTensor.cuh"
#include <exception>
#include <numeric>
#include "../Utilities.h"
#include <functional>

namespace DeepLearning
{
	namespace CostFunctionHelper
	{
		Real evaluate_cost(const BasicCollection& output, const BasicCollection& reference, const CostFunctionId id)
		{
			const auto func = CostFunctionHelper::make<std::function<Real(Real, Real)>>(id);

			const auto result = std::transform_reduce(output.begin(), output.end(), reference.begin(), Real(0), std::plus<Real>(),
				[&](const auto& x, const auto& ref) { return  func(x, ref); });
			return result;
		}

		Real evaluate_cost_and_gradient(BasicCollection& output, const BasicCollection& reference, const CostFunctionId id)
		{
			auto func_val = Real(0);

			const auto func = CostFunctionHelper::make<std::function<dual<Real>(dual<Real>, Real)>>(id);

			for (auto item_id = 0ull; item_id < output.size(); item_id++)
			{
				const auto res = func({ output.begin()[item_id] , Real(1) }, reference.begin()[item_id]);
				func_val += res.Real();
				output.begin()[item_id] = res.Dual()[0];
			}

			return func_val;
		}

		void evaluate_gradient(BasicCollection& output, const BasicCollection& reference, const CostFunctionId id)
		{
			const auto func = CostFunctionHelper::make<std::function<dual<Real>(dual<Real>, Real)>>(id);
			std::transform(output.begin(), output.end(), reference.begin(), output.begin(), [&](const auto& x, const auto& ref) {
				return  func({ x, Real(1) }, ref).Dual()[0]; });
		}
	}

	template <class T>
	CostFunction<T>::CostFunction(const CostFunctionId id) : _id(id)
	{}

	template <class T>
	Real CostFunction<T>::operator ()(const T& output, const T& reference) const
	{
		if (output.size() != reference.size())
			throw std::exception("Incompatible input");

		return CostFunctionHelper::evaluate_cost(output, reference, _id);
	}

	template <class T>
	std::tuple<Real, T> CostFunction<T>::func_and_deriv(const T& output, const T& reference) const
	{
		if (output.size() != reference.size())
			throw std::exception("Incompatible input");

		auto deriv = output;
		const auto func_val = CostFunctionHelper::evaluate_cost_and_gradient(deriv, reference, _id);
		return std::make_tuple(func_val, std::move(deriv));
	}

	template <class T>
	void CostFunction<T>::deriv_in_place(T& output_deriv, const T& reference) const
	{
		CostFunctionHelper::evaluate_gradient(output_deriv, reference, _id);
	}

	template <class T>
	T CostFunction<T>::deriv(const T& output, const T& reference) const
	{
		auto deriv = output;
		deriv_in_place(deriv, reference);
		return deriv;
	}

	std::string to_string(const CostFunctionId& cost_type_id)
	{
		switch (cost_type_id)
		{
		case CostFunctionId::SQUARED_ERROR: return "SQUARED_ERROR";
		case CostFunctionId::CROSS_ENTROPY: return "CROSS_ENTROPY";
		default:
			return "UNKNOWN";
		}
	}

	CostFunctionId parse_cost_type(const std::string& str)
	{
		const auto str_normalized = Utils::normalize_string(str);

		for (unsigned int id = (unsigned int)CostFunctionId::SQUARED_ERROR; id <= (unsigned int)CostFunctionId::CROSS_ENTROPY; id++)
		{
			if (to_string((CostFunctionId)id) == str_normalized)
				return (CostFunctionId)id;
		}

		return CostFunctionId::UNKNOWN;
	}

	template class CostFunction<Vector>;
	template class CostFunction<CudaVector>;
	template class CostFunction<Matrix>;
	template class CostFunction<CudaMatrix>;
	template class CostFunction<Tensor>;
	template class CostFunction<CudaTensor>;
}