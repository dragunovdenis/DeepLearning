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

#include "../defs.h"
#include <cuda_runtime.h>


namespace DeepLearning
{
#define gpuErrchk(ans) { CudaUtils::gpuAssert((ans), __FILE__, __LINE__); }

#ifdef CUDA_DEBUG
#define CUDA_SANITY_CHECK { CudaUtils::gpuAssert((cudaDeviceSynchronize()), __FILE__, __LINE__); }
#else
#define CUDA_SANITY_CHECK
#endif

	namespace CudaUtils
	{
		/// <summary>
		/// Method to process CUDA related exit codes
		/// </summary>
		void gpuAssert(cudaError_t code, const char* file, int line);

		/// <summary>
		/// Allocates given number of elements of type "T" in the "device" memory
		/// </summary>
		template <class T>
		T* cuda_allocate(const std::size_t& size)
		{
			void* temp;
			gpuErrchk(cudaMallocAsync(&temp, size * sizeof(T), cudaStreamPerThread));

			return static_cast<T*>(temp);
		}

		/// <summary>
		/// Frees the pointed device memory
		/// </summary>
		void cuda_free(void* devPtr);

		/// <summary>
		/// Utility method to do memory copying
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="dest">Pointer to destination memory</param>
		/// <param name="src">Pointer to source memory</param>
		/// <param name="size">Number of items of size "T" to be copied</param>
		/// <param name="copy_kind">Copy "direction"</param>
		template <class T>
		void cuda_copy(T* dest, const T* src, const std::size_t& size, const cudaMemcpyKind& copy_kind)
		{
			gpuErrchk(cudaMemcpyAsync(dest, src, size * sizeof(T), copy_kind, cudaStreamPerThread));
		}

		/// <summary>
		/// "Checked" version of CUDA device-to-device memory copying
		/// </summary>
		template <class T>
		void cuda_copy_device2device(T* dest, const T* src, const std::size_t& size)
		{
			cuda_copy(dest, src, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
		}

		/// <summary>
		/// "Checked" version of CUDA device-to-host memory copying
		/// </summary>
		template <class T>
		void cuda_copy_device2host(T* dest, const T* src, const std::size_t& size)
		{
			cuda_copy(dest, src, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		}

		/// <summary>
		/// "Checked" version of CUDA host-to-device memory copying
		/// </summary>
		template <class T>
		void cuda_copy_host2device(T* dest, const T* src, const std::size_t& size)
		{
			cuda_copy(dest, src, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
		}

		/// <summary>
		/// Fills given array of elements of type "T" with zeros
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="arr">Pinter to the first element of the given array</param>
		/// <param name="size">Size of array (number of elements of type "T" in the array)</param>
		template <class T>
		void fill_zero(T* arr, const std::size_t& size)
		{
			gpuErrchk(cudaMemsetAsync(arr, 0, size * sizeof(T), cudaStreamPerThread));
		}
	}
}