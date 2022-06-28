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

#pragma once
#include "ConvolutionUtils.h"

namespace DeepLearning::ConvolutionUtils
{
	/// <summary>
	/// Returns size of convolution result in certain dimension
	/// </summary>
	/// <param name="in_size">Input size (in the chosen dimension)</param>
	/// <param name="kernel_size">Kernel size (in the chosen dimension)</param>
	/// <param name="padding">Padding size (in the chosen dimension)</param>
	/// <param name="stride">Stride size (in the chosen dimension)</param>
	long long calc_out_size_for_convolution(const std::size_t in_size, const std::size_t kernel_size,
		const long long padding, const long long stride)
	{
		const auto temp = static_cast<long long>(in_size) + 2 * padding - static_cast<long long>(kernel_size);
		if (temp < 0)
			return 0;

		return  temp / stride + 1ull;
	}

	Index3d calc_conv_res_size(const Index3d& tensor_size, const Index3d& kernel_size, const Index3d& paddings, const Index3d& strides)
	{
		return { calc_out_size_for_convolution(tensor_size.x, kernel_size.x, paddings.x, strides.x),
				 calc_out_size_for_convolution(tensor_size.y, kernel_size.y, paddings.y, strides.y),
				 calc_out_size_for_convolution(tensor_size.z, kernel_size.z, paddings.z, strides.z) };
	}
}
