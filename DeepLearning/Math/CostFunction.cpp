#include "CostFunction.h"
#include "DenseVector.h"
#include <exception>
#include <numeric>

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
				return -(ref*log(x) + (Real(1) - ref)*log(Real(1) - x));
			});
			break;
		default: throw std::exception("Unexpected cost function ID.");
			break;
		}
	}

	Real CostFunction::operator ()(const DenseVector& output, const DenseVector& reference) const
	{
		if (output.dim() != reference.dim())
			throw std::exception("Incompatible input");

		const auto result = std::transform_reduce(output.begin(), output.end(), reference.begin(), Real(0), std::plus<Real>(),
			[&](const auto& x, const auto& ref) { return  _func->operator()(x, ref); });
		return result;
	}

	std::tuple<Real, DenseVector> CostFunction::func_and_deriv(const DenseVector& output, const DenseVector& reference) const
	{
		if (output.dim() != reference.dim())
			throw std::exception("Incompatible input");

		DenseVector deriv(output.dim());
		auto func_val = Real(0);

		for (std::size_t item_id = 0; item_id < output.dim(); item_id++)
		{
			const auto [value, derivative] = _func->calc_funcion_and_derivative(output(item_id), reference(item_id));
			func_val += value;
			deriv(item_id) = derivative;
		}

		return std::make_tuple(func_val, deriv);
	}
}