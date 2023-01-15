//Copyright (c) 2023 Denys Dragunov, dragunovdenis@gmail.com
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

#include "LayerGradient.h"
#include "DataContext.h"
#include "../Math/CollectionArithmetics.h"

namespace DeepLearning
{
	template <class D>
	bool LayerGradient<D>::operator ==(const LayerGradient& lg) const
	{
		return Biases_grad == lg.Biases_grad && Weights_grad == lg.Weights_grad;
	}

	template <class D>
	bool LayerGradient<D>::operator !=(const LayerGradient& lg) const
	{
		return !(*this == lg);
	}

	template <class D>
	LayerGradient<D>& LayerGradient<D>::operator +=(const LayerGradient& lg)
	{
		Biases_grad += lg.Biases_grad;
		Weights_grad += lg.Weights_grad;
		return *this;
	}

	template <class D>
	LayerGradient<D> operator +(const LayerGradient<D>& lg, const LayerGradient<D>& lg1)
	{
		auto result = lg;
		return result += lg1;
	}

	template LayerGradient<CpuDC> operator +(const LayerGradient<CpuDC>& lg, const LayerGradient<CpuDC>& lg1);
	template LayerGradient<GpuDC> operator +(const LayerGradient<GpuDC>& lg, const LayerGradient<GpuDC>& lg1);

	template <class D>
	LayerGradient<D>& LayerGradient<D>::operator -=(const LayerGradient& lg)
	{
		Biases_grad -= lg.Biases_grad;
		Weights_grad -= lg.Weights_grad;
		return *this;
	}

	template <class D>
	LayerGradient<D> operator -(const LayerGradient<D>& lg, const LayerGradient<D>& lg1)
	{
		auto result = lg;
		return result -= lg1;
	}

	template  LayerGradient<CpuDC> operator -(const LayerGradient<CpuDC>& lg, const LayerGradient<CpuDC>& lg1);
	template  LayerGradient<GpuDC> operator -(const LayerGradient<GpuDC>& lg, const LayerGradient<GpuDC>& lg1);

	template <class D>
	LayerGradient<D>& LayerGradient<D>::operator *=(const Real& scalar)
	{
		Biases_grad *= scalar;
		Weights_grad *= scalar;
		return *this;
	}

	template <class D>
	LayerGradient<D> operator *(const LayerGradient<D>& lg, const Real& scalar)
	{
		auto result = lg;
		return result *= scalar;
	}

	template LayerGradient<CpuDC> operator *(const LayerGradient<CpuDC>& lg, const Real& scalar);
	template LayerGradient<GpuDC> operator *(const LayerGradient<GpuDC>& lg, const Real& scalar);

	template <class D>
	LayerGradient<D> operator *(const Real& scalar, const LayerGradient<D>& lg)
	{
		return lg * scalar;
	}

	template LayerGradient<CpuDC> operator *(const Real& scalar, const LayerGradient<CpuDC>& lg);
	template LayerGradient<GpuDC> operator *(const Real& scalar, const LayerGradient<GpuDC>& lg);

	template <class D>
	Real LayerGradient<D>::max_abs() const
	{
		auto result = static_cast<Real>(0);

		for (auto item_id = 0ull; item_id < Weights_grad.size(); ++item_id)
		{
			const auto wm_abs = Weights_grad[item_id].max_abs();
			if (result < wm_abs)
				result = wm_abs;
		}

		const auto bm_abs = Biases_grad.max_abs();
		if (result < bm_abs)
			result = bm_abs;

		return result;
	}

	template <class D>
	bool LayerGradient<D>::empty() const
	{
		return Weights_grad.empty() && Biases_grad.empty();
	}

	template struct LayerGradient<CpuDC>;
	template struct LayerGradient<GpuDC>;
}