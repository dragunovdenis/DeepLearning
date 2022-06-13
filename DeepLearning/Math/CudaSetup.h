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

#include "cuda_runtime.h"
#include <cstddef>

namespace DeepLearning
{
	/// <summary>
	/// Functionality to perform basic initialization of CUDA-related properties
	/// Including choosing of the most performing devise
	/// </summary>
	class CudaSetup
	{
		/// <summary>
		/// Initialization status
		/// </summary>
		cudaError_t _status = cudaError_t::cudaErrorAssert;

		/// <summary>
		/// Properties of the chosen device (provided that initialization was successful)
		/// </summary>
		cudaDeviceProp _device_props{};

		/// <summary>
		/// Private constructor to implement a singleton pattern
		/// </summary>
		CudaSetup();

		static CudaSetup _instance;

	public:

		/// <summary>
		/// Returns "true" if the initialization was successful
		/// </summary>
		static bool is_successful();

		/// <summary>
		/// Returns initialization status
		/// </summary>
		static cudaError_t status();

		/// <summary>
		/// Returns maximal allowed number of threads per block for the chosen device (provided that the initialization went successful) 
		/// </summary>
		static std::size_t threads_per_block();

		/// <summary>
		/// Calculates number of blocks needed to launch a 1D kernel for the given number of items
		/// </summary>
		static std::size_t calc_blocks(const std::size_t& items_cnt);
	};
}