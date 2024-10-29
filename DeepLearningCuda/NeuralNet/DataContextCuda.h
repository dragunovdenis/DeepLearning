//Copyright (c) 2024 Denys Dragunov, dragunovdenis@gmail.com
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
#include "../Math/CudaVector.cuh"
#include "../Math/CudaMatrix.cuh"
#include "../Math/CudaTensor.cuh"
#include "../Math/CudaArray.cuh"
#include <NeuralNet/DataContext.h>

namespace DeepLearning
{
	/// <summary>
	/// A set of data types suitable for conducting calculations on GPU 
	/// </summary>
	class GpuDC
	{
	public:
		using vector_t = CudaVector;
		using matrix_t = CudaMatrix;
		using tensor_t = CudaTensor;

		template <typename I>
		using index_array_t = CudaArray<I>;

		/// <summary>
		/// Converter.
		/// </summary>
		static CpuDC::vector_t to_host(const vector_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static CpuDC::matrix_t to_host(const matrix_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static CpuDC::tensor_t to_host(const tensor_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static vector_t from_host(const CpuDC::vector_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static matrix_t from_host(const CpuDC::matrix_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static tensor_t from_host(const CpuDC::tensor_t& source);
	};
}
