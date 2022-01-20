#include "AuxLearaningData.h"
#include <exception>

namespace DeepLearning
{
	CummulativeLayerGradient::CummulativeLayerGradient(const std::size_t in_dim, const std::size_t out_dim)
	{
		_sum_grad_weights = DenseMatrix(out_dim, in_dim);
		_sum_grad_biases = DenseVector(out_dim);
	}

	void CummulativeLayerGradient::Add(const DenseMatrix& weight_grad, const DenseVector& bias_grad)
	{
		_sum_grad_weights += weight_grad;
		_sum_grad_biases += bias_grad;
		_accumulated_items_count++;
	}

	std::tuple<DenseMatrix, DenseVector> CummulativeLayerGradient::calc_average_grarient(const Real scale_factor) const
	{
		if (_accumulated_items_count == 0)
			throw std::exception("No items have been added.");

		const auto factor = scale_factor / _accumulated_items_count;
		return std::make_tuple(_sum_grad_weights * factor, _sum_grad_biases * factor);
	}

	void CummulativeLayerGradient::reset()
	{
		_sum_grad_biases.fill(Real(0));
		_sum_grad_weights.fill(Real(0));
		_accumulated_items_count = 0;
	}
}