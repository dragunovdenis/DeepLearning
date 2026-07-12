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
#include "../Math/Vector.h"
#include "../Math/Matrix.h"
#include "../Math/Tensor.h"

namespace DeepLearning
{
	/// <summary>
	/// A set of data types suitable for conducting calculations on CPU 
	/// </summary>
	class CpuDC
	{
	public:
		using vector_t = Vector;
		using matrix_t = Matrix;
		using tensor_t = Tensor;
		using basic_collection_t = BasicCollection;

		template <typename I>
		using index_array_t = std::vector<I>;

		/// <summary>
		/// Converter.
		/// </summary>
		static vector_t to_host(const vector_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static matrix_t to_host(const matrix_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static tensor_t to_host(const tensor_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static vector_t from_host(const vector_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static matrix_t from_host(const matrix_t& source);

		/// <summary>
		/// Converter.
		/// </summary>
		static tensor_t from_host(const tensor_t& source);

		/// <summary>
		/// Tag-field to indicate that the corresponding data
		/// structures within the current context support fast
		/// convolution algorithms.
		/// </summary>
		static bool constexpr  supports_fast_convolution() { return true; }
	};
}
