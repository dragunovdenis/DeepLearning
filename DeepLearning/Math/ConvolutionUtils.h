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
#include "LinAlg3d.h"

namespace DeepLearning::ConvolutionUtils
{
#define KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets, action)				\
			for (auto k_x = kernel_start_offsets.x; k_x < kernel_stop_offsets.x; k_x++)				\
			{																						\
				const auto t_x = tensor_offsets.x + k_x;											\
				for (auto k_y = kernel_start_offsets.y; k_y < kernel_stop_offsets.y; k_y++)			\
				{																					\
					const auto t_y = tensor_offsets.y + k_y;										\
					for (auto k_z = kernel_start_offsets.z; k_z < kernel_stop_offsets.z; k_z++)		\
					{																				\
						const auto t_z = tensor_offsets.z + k_z;									\
						action;																		\
					}																				\
				}																					\
			}

	/// <summary>
	/// Data structure to hold offsets used when calculating convolution
	/// </summary>
	struct offset_data
	{
		Index3d tensor_offsets;
		Index3d kernel_start_offsets;
		Index3d kernel_stop_offsets;
	};

	/// <summary>
	/// Converts given index of an element in the "data" array to a triplet of layer, row and column indices of the same element
	/// </summary>
	/// <param name="data_id">Index of an element in the "data" array</param>
	/// <param name="tensor_size">3d size of the tensor</param>
	inline CUDA_CALLABLE Index3d data_id_to_index_3d(const long long data_id, const Index3d& tensor_size)
	{
		auto temp = data_id / tensor_size.z;
		const auto col_id = data_id % tensor_size.z;
		const auto layer_id = temp / tensor_size.y;
		const auto row_id = temp % tensor_size.y;

		return { layer_id, row_id, col_id };
	}

	/// <summary>
	/// Calculates offsets needed to calculate convolution result item with the given offsets (3d index)
	/// </summary>
	/// <param name="conv_res_offsets">Offset (3d index) of the convolution result item</param>
	/// <param name="tensor_size">Size of the tensor the convolution is applied to</param>
	/// <param name="kernel_size">Size of the convolution kernel</param>
	/// <param name="paddings">Zero paddings to be applied to the tensor</param>
	/// <param name="strides">Strides to be used when shifting convolution kernel over the tensor</param>
	/// <returns>Tuple consisting of "tensor_offsets, kernel_start_offsets, kernel_stop_offsets" in the exact same order</returns>
	inline CUDA_CALLABLE offset_data calc_kernel_loop_offsets(const Index3d& conv_res_offsets, const Index3d& tensor_size,
		const Index3d& kernel_size, const Index3d& paddings, const Index3d& strides)
	{
		const auto tensor_offsets = conv_res_offsets.hadamard_prod(strides) - paddings;
		const auto kernel_start_offsets = relu(-tensor_offsets);
		const auto kernel_stop_offsets = kernel_size - relu(tensor_offsets + kernel_size - tensor_size);

		return { tensor_offsets, kernel_start_offsets, kernel_stop_offsets };
	}

	/// <summary>
	/// Returns total size of convolution result
	/// </summary>
	/// <param name="tensor_size">Size of the tensor the convolution is to be applied to</param>
	/// <param name="kernel_size">Size of the convolution kernel</param>
	/// <param name="paddings">Sizes of zero paddings of the tensor</param>
	/// <param name="strides">Sizes of strides to be used</param>
	Index3d calc_conv_res_size(const Index3d& tensor_size, const Index3d& kernel_size, const Index3d& paddings, const Index3d& strides);
}