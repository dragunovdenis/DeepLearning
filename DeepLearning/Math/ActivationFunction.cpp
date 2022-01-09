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

	DenseVector Sigmoid::operator ()(const DenseVector& input) const
	{
		DenseVector result(input.dim());
		std::transform(input.begin(), input.end(), result.begin(),
			[](const auto& x) { return sigmoid(x); });
		return result;
	}

	std::tuple<DenseVector, DenseVector> Sigmoid::func_and_deriv(const DenseVector& input) const
	{
		std::vector<dual<Real>> temp(input.dim());

		std::transform(input.begin(), input.end(), temp.begin(),
			[](const auto& x)
			{
				return sigmoid(dual<Real>{x, { Real(1) }});
			});

		DenseVector func(input.dim());
		DenseVector deriv(input.dim());

		for (std::size_t item_id = 0; item_id < input.dim(); item_id++)
		{
			const auto& item = temp[item_id];
			func(item_id) = item.Real();
			deriv(item_id) = item.Dual()[0];
		}

		return std::make_tuple(func, deriv);
	}
}