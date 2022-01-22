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