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

#include "DataContextCuda.h"

namespace DeepLearning
{
	CpuDC::vector_t GpuDC::to_host(const vector_t& source)
	{
		return source.to_host();
	}

	CpuDC::matrix_t GpuDC::to_host(const matrix_t& source)
	{
		return source.to_host();
	}

	CpuDC::tensor_t GpuDC::to_host(const tensor_t& source)
	{
		return source.to_host();
	}

	GpuDC::vector_t GpuDC::from_host(const CpuDC::vector_t& source)
	{
		return vector_t(source);
	}

	GpuDC::matrix_t GpuDC::from_host(const CpuDC::matrix_t& source)
	{
		return matrix_t(source);
	}

	GpuDC::tensor_t GpuDC::from_host(const CpuDC::tensor_t& source)
	{
		return tensor_t(source);
	}
}