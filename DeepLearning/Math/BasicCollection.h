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
#include <functional>

namespace DeepLearning
{
	/// <summary>
	/// "Basic" extension that can be useful for a collection-class
	/// </summary>
	class BasicCollection
	{
	protected:

		/// <summary>
		/// Size of the vector
		/// </summary>
		virtual std::size_t size() const = 0;

		/// <summary>
		/// Element-wise sum with another collection of the same size
		/// It is a responsibility of the caller to make sure that the collections are of the same size
		/// </summary>
		void add(const BasicCollection& collection);

		/// <summary>
		/// Element-wise difference with another collection of the same size
		/// It is a responsibility of the caller to make sure that the collections are of the same size
		/// </summary>
		void sub(const BasicCollection& collection);

		/// <summary>
		/// Multiplies each element of the current collection by the given scalar
		/// </summary>
		void mul(const Real& scalar);

	public:

		/// <summary>
		/// Pointer to the first element of the vector
		/// </summary>
		virtual Real* begin() = 0;

		/// <summary>
		/// Pointer to the first element of the vector (constant version)
		/// </summary>
		virtual const Real* begin() const = 0;

		/// <summary>
		/// Pointer to the "behind last" element of the vector
		/// </summary>
		virtual Real* end() = 0;

		/// <summary>
		/// Pointer to the "behind last" element of the vector (constant version)
		/// </summary>
		virtual const Real* end() const = 0;

		/// <summary>
		/// "Maximal absolute value" norm ("infinity" norm) of the collection
		/// </summary>
		Real max_abs() const;

		/// <summary>
		/// Returns sum of all the elements of the collection transformed with the given operator
		/// </summary>
		Real sum(const std::function<Real(Real)>& transform_operator = [](const auto& x) {return x; }) const;

		/// <summary>
		/// Assigns the given value to all the elements of the collection
		/// </summary>
		void fill(const Real& val);

		/// <summary>
		/// Returns "true" if the collection is empty
		/// </summary>
		bool empty() const;
	};

}
