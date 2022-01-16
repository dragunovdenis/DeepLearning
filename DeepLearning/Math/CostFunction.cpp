#include "CostFunction.h"
#include "DenseVector.h"
#include <exception>

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

	DenseVector CostFunction::operator ()(const DenseVector& output, const DenseVector& reference) const
	{
		if (output.dim() != reference.dim())
			throw std::exception("Incompatible input");

		DenseVector result(output.dim());
		std::transform(output.begin(), output.end(), reference.begin(), result.begin(),
			[&](const auto& x, const auto& ref) { return  _func->operator()(x, ref); });
		return result;
	}

	std::tuple<DenseVector, DenseVector> CostFunction::func_and_deriv(const DenseVector& output, const DenseVector& reference) const
	{
		if (output.dim() != reference.dim())
			throw std::exception("Incompatible input");

		DenseVector func(output.dim());
		DenseVector deriv(output.dim());

		for (std::size_t item_id = 0; item_id < output.dim(); item_id++)
		{
			const auto [value, derivative] = _func->calc_funcion_and_derivative(output(item_id), reference(item_id));
			func(item_id) = value;
			deriv(item_id) = derivative;
		}

		return std::make_tuple(func, deriv);
	}
}