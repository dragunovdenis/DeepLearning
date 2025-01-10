//Copyright (c) 2023 Denys Dragunov, dragunovdenis@gmail.com
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
#include <vector>
#include <msgpack.hpp>
#include "../defs.h"

namespace DeepLearning
{
	/// <summary>
	/// A data structure to hold output of the back-propagation procedure
	/// </summary>
	template <class D>
	struct LayerGradient
	{
		/// <summary>
		/// Container to hold generalized gradient.
		/// </summary>
		std::vector<typename D::tensor_t> data{};

		MSGPACK_DEFINE(data);

		/// <summary>
		///	Equality operator
		/// </summary>
		bool operator ==(const LayerGradient& lg) const;

		/// <summary>
		///	Inequality operator
		/// </summary>
		bool operator !=(const LayerGradient& lg) const;

		/// <summary>
		///	Compound addition operator
		/// </summary>
		LayerGradient& operator +=(const LayerGradient& lg);

		/// <summary>
		///	Compound subtraction operator
		/// </summary>
		LayerGradient& operator -=(const LayerGradient& lg);

		/// <summary>
		///	Compound multiplication by scalar operator
		/// </summary>
		LayerGradient& operator *=(const Real& scalar);

		/// <summary>
		///	L-infinity norm
		/// </summary>
		[[nodiscard]] Real max_abs() const;

		/// <summary>
		///	Returns "true" if the instance is trivial (default assigned)
		/// </summary>
		[[nodiscard]] bool empty() const;

		/// <summary>
		/// Adds the given gradient instance multiplied by the given scalar to the current gradient instance
		/// </summary>
		LayerGradient& add_scaled(const LayerGradient& lg, const Real& scalar);

		/// <summary>
		/// Scales the current instance of gradient by the given scalar and adds the given instance of gradient to it
		/// </summary>
		LayerGradient& scale_and_add(const Real& scalar, const LayerGradient& lg);

		/// <summary>
		/// Fills all the sub-collections with zeros.
		/// </summary>
		void fill_zero();
	};

	/// <summary>
	///	Multiplication by scalar operator
	/// </summary>
	template <class D>
	LayerGradient<D> operator *(const LayerGradient<D>& lg, const Real& scalar);

	/// <summary>
	///	Multiplication by scalar operator
	/// </summary>
	template <class D>
	LayerGradient<D> operator *(const Real& scalar, const LayerGradient<D>& lg);

	/// <summary>
	///	Addition operator
	/// </summary>
	template <class D>
	LayerGradient<D> operator +(const LayerGradient<D>& lg, const LayerGradient<D>& lg1);

	/// <summary>
	///	Subtraction operator
	/// </summary>
	template <class D>
	LayerGradient<D> operator -(const LayerGradient<D>& lg, const LayerGradient<D>& lg1);
}

#include "LayerGradient.inl"
