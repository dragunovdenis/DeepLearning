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
#include "DataContext.h"

namespace DeepLearning
{
	/// <summary>
	/// Auxiliary structure to hold data that is produced during an
	/// "act" phase and can be used during the subsequent "backpropagation" phase.
	/// </summary>
	template <class D = CpuDC>
	struct LayerTraceData
	{
		/// <summary>
		/// Container to store derivatives of the activation function.
		/// </summary>
		typename D::tensor_t Derivatives{};

		/// <summary>
		/// Container for index mappings.
		/// </summary>
		typename D::template index_array_t<int> IndexMapping{};
	};

	/// <summary>
	/// Container to store processing-related data
	/// (e.g. input data as well as data needed to run backpropagation) for a layer.
	/// </summary>
	template <class D = CpuDC>
	struct LayerData
	{
		/// <summary>
		/// Container to store input data for a layer.
		/// </summary>
		typename D::tensor_t Input{};

		/// <summary>
		/// Trace data.
		/// </summary>
		LayerTraceData<D> Trace{};

		/// <summary>
		/// Default constructor.
		/// </summary>
		LayerData() = default;

		/// <summary>
		/// Constructor.
		/// </summary>
		LayerData(const typename D::tensor_t& input) : Input(input){}
	};
}
