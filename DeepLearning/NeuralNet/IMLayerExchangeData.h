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

namespace DeepLearning
{
	/// <summary>
	/// Facade to work with input-output data of m-layers,
	/// i.e. data that can be exchanged between layers,
	/// as an input for the next layer would be the output for the previous layer.
	/// </summary>
	template <class T>
	struct IMLayerExchangeData
	{
		/// <summary>
		/// Virtual destructor.
		/// </summary>
		virtual ~IMLayerExchangeData() = default;

		/// <summary>
		/// Number of items in the basic collection.
		/// </summary>
		virtual std::size_t size() const = 0;

		/// <summary>
		/// Resizes the underlying collection of items.
		/// </summary>
		virtual void resize(const std::size_t new_size) = 0;

		/// <summary>
		/// Access to a storage allowing to share processing data between (adjacent) m-layers.
		/// </summary>
		virtual const T& operator[](const std::size_t layer_id) const = 0;

		/// <summary>
		/// Access to a storage allowing to share processing data between (adjacent) m-layers. Const version.
		/// </summary>
		virtual T& operator[](const std::size_t layer_id) = 0;
	};
}