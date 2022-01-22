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

#include <vector>
#include "../defs.h"
#include <msgpack.hpp>
#include <functional>

namespace DeepLearning
{
	/// <summary>
	/// Represents a dense vector of arbitrary dimension
	/// </summary>
	class DenseVector
	{
		std::vector<Real> _data{};

		/// <summary>
		/// Returns "true" if the given index is valid (within the boundaries of the underlying "data" collection)
		/// </summary>
		bool check_bounds(const ::std::size_t id) const;

	public:

		MSGPACK_DEFINE(_data);

		/// <summary>
		/// Default constructor
		/// </summary>
		DenseVector() = default;

		/// <summary>
		/// Constructs dense vector of the given dimension
		/// </summary>
		DenseVector(const std::size_t dim);

		/// <summary>
		/// Constructor from given source vector
		/// </summary>
		template <class T>
		DenseVector(const std::vector<T>& source);

		/// <summary>
		/// Constructs dense vector of the given dimension filled with
		/// uniformly distributed pseudo-random values from the given range
		/// </summary>
		DenseVector(const std::size_t dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// Returns dimension of the vector
		/// </summary>
		std::size_t dim() const;

		/// <summary>
		/// Element access operator
		/// </summary>
		/// <param name="id">Index of the element to be accessed</param>
		/// <returns>Reference to the element</returns>
		Real& operator ()(const std::size_t id);

		/// <summary>
		/// Element access operator (constant version)
		/// </summary>
		/// <param name="id">Index of the element to be accessed</param>
		/// <returns>Constant reference to the element</returns>
		const Real& operator ()(const std::size_t id) const;

		/// <summary>
		/// Compound addition operator
		/// </summary>
		DenseVector& operator += (const DenseVector& vec);

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		DenseVector& operator -= (const DenseVector& vec);

		/// <summary>
		/// Compound scalar multiplication operator
		/// </summary>
		DenseVector& operator *= (const Real& scalar);

		/// <summary>
		/// Equality operator
		/// </summary>
		bool operator ==(const DenseVector& vect) const;

		/// <summary>
		/// Inequality operator
		/// </summary>
		bool operator !=(const DenseVector& vect) const;

		/// <summary>
		/// Iterator pointing to the first element of the vector
		/// </summary>
		std::vector<Real>::iterator begin();

		/// <summary>
		/// Iterator pointing to the first element of the vector (constant version)
		/// </summary>
		std::vector<Real>::const_iterator begin() const;

		/// <summary>
		/// Iterator pointing to the "behind last" element of the vector
		/// </summary>
		std::vector<Real>::iterator end();

		/// <summary>
		/// Iterator pointing to the "behind last" element of the vector (constant version)
		/// </summary>
		std::vector<Real>::const_iterator end() const;

		/// <summary>
		/// Generates a vector filled with uniformly distributed pseudo random values
		/// </summary>
		static DenseVector random(const std::size_t dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// "Maximal absolute value" norm ("infinity" norm)
		/// </summary>
		Real max_abs() const;

		/// <summary>
		/// Returns index of the "maximal element" defined by the given comparer
		/// or "1" if the vector contains zero elements
		/// </summary>
		std::size_t max_element_id(const std::function<bool(Real, Real)>&comparer = [](const auto& a, const auto& b) {return a < b; }) const;

		/// <summary>
		/// Returns sum of all the elements of the vector transformed with the given operator
		/// </summary>
		Real sum(const std::function<Real(Real)>& transform_operator = [](const auto& x) {return x; }) const;

		/// <summary>
		/// Assigns the given value to all the elements of the vector
		/// </summary>
		void fill(const Real & val);

		/// <summary>
		/// Returns Hadamard (element-wise) product of the current vector with the input
		/// </summary>
		DenseVector hadamard_prod(const DenseVector & vec) const;
	};

	/// <summary>
	/// Vector addition operator
	/// </summary>
	DenseVector operator + (const DenseVector& vec1, const DenseVector& vec2);

	/// <summary>
	/// Vector subtraction operator
	/// </summary>
	DenseVector operator -(const DenseVector& vec1, const DenseVector& vec2);

	/// <summary>
	/// Vector-Scalar multiplication operator
	/// </summary>
	DenseVector operator *(const DenseVector& vec, const Real& scalar);

	/// <summary>
	/// Scalar-vector multiplication operator
	/// </summary>
	DenseVector operator *(const Real& scalar, const DenseVector& vec);
}
