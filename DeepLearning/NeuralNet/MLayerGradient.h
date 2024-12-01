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
#include <msgpack.hpp>
#include "LayerGradient.h"

namespace DeepLearning
{
	/// <summary>
	/// Container to hold gradient of an m-layer with respect to its parameters.
	/// </summary>
	template <class D>
	struct MLayerGradient
	{
	private:

		std::vector<LayerGradient<D>> sub_gradients{};

	public:

		/// <summary>
		/// Subscript operator.
		/// </summary>
		LayerGradient<D>& operator [](const int id)
		{
			return sub_gradients[id];
		}

		/// <summary>
		/// Subscript operator (const version).
		/// </summary>
		const LayerGradient<D>& operator [](const int id) const
		{
			return sub_gradients[id];
		}

		/// <summary>
		/// Returns size of the collection.
		/// </summary>
		std::size_t size() const
		{
			return sub_gradients.size();
		}

		/// <summary>
		/// Default constructor.
		/// </summary>
		MLayerGradient() = default;

		/// <summary>
		/// Constructor.
		/// </summary>
		MLayerGradient(const int size) : sub_gradients(size) {}

		MSGPACK_DEFINE(sub_gradients);
	};
}
