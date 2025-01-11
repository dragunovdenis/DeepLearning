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
#include "LazyVector.h"

namespace DeepLearning
{
	/// <summary>
	/// Auxiliary data structure used for more efficient memory usage when evaluating networks.
	/// </summary>
	template <class D = CpuDC>
	struct InOutMData
	{
	private:
		LazyVector<typename D::tensor_t> _data[2];
		int _id = 0;

		/// <summary>
		/// Returns next id for the swapping functionality.
		/// </summary>
		int next_id() const
		{
			return _id + 1 & 1;
		}
	public:
		/// <summary>
		/// Current input container for a layer.
		/// </summary>
		LazyVector<typename D::tensor_t>& in()
		{
			return _data[_id];
		}

		/// <summary>
		/// Current container to receive output of a layer.
		/// </summary>
		LazyVector<typename D::tensor_t>& out()
		{
			return _data[next_id()];
		}

		/// <summary>
		/// Current container to receive output of a layer (readonly version).
		/// </summary>
		const LazyVector<typename D::tensor_t>& out() const
		{
			return _data[next_id()];
		}

		/// <summary>
		/// Swaps input and output data structs.
		/// </summary>
		void swap()
		{
			_id = next_id();
		}
	};
}
