#include "DiffFunc.h"

namespace DeepLearning
{
	Real DiffFunc::operator()(const Real& arg, const Real& param) const
	{
		return func(arg, param);
	}

	std::tuple<Real, Real> DiffFunc::calc_funcion_and_derivative(const Real arg, const Real& param) const
	{
		const auto result_dual = func_dual({ arg, {Real(1)} }, param);
		return std::make_tuple(result_dual.Real(), result_dual.Dual()[0]);
	}
}