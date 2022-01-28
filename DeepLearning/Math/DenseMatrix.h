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
#include <msgpack.hpp>
#include <functional>
#include "../defs.h"

namespace DeepLearning
{
	class DenseVector;

	/// <summary>
	/// Representation of a dense rectangular matrix
	/// </summary>
	class DenseMatrix
	{
		/// <summary>
		/// Elements of the matrix in a "flattened" form
		/// </summary>
		std::vector<Real> _data{};

		/// <summary>
		/// Matrix dimensions
		/// </summary>
		std::size_t _row_dim{};
		std::size_t _col_dim{};

		/// <summary>
		/// Size of the inner array
		/// </summary>
		std::size_t size() const;

		/// <summary>
		/// Returns true if the given row and column indices are valid
		/// </summary>
		bool check_bounds(const std::size_t row_id, const std::size_t col_id) const;

		/// <summary>
		/// Converts the given row and column indices into a single index in the data collection
		/// The caller is responsible for index validation
		/// </summary>
		/// <param name="row_id">Row index</param>
		/// <param name="col_id">Column index</param>
		std::size_t row_col_to_data_id(const std::size_t row_id, const std::size_t col_id) const;

	public:

		MSGPACK_DEFINE(_row_dim, _col_dim, _data);

		/// <summary>
		/// Column dimension getter
		/// </summary>
		std::size_t col_dim() const;

		/// <summary>
		/// Row dimension getter
		/// </summary>
		std::size_t row_dim() const;

		/// <summary>
		/// Default constructor, constructs an empty matrix (i.e. a matrix having "zero" dimensions)
		/// </summary>
		DenseMatrix() = default;

		/// <summary>
		/// Constructs a dense matrix of the given dimensions
		/// </summary>
		DenseMatrix(const std::size_t row_dim, const std::size_t col_dim);

		/// <summary>
		/// Constructs a dense matrix of the given dimensions filled according to the given generator function
		/// </summary>
		DenseMatrix(const std::size_t row_dim, const std::size_t col_dim, const std::function<Real()>& generator);

		/// <summary>
		/// Constructs a dense matrix of the given dimensions filled
		/// with a uniformly distributed pseudo-random values from the given range
		/// </summary>
		DenseMatrix(const std::size_t row_dim, const std::size_t col_dim,
			const Real range_begin, const Real range_end);

		/// <summary>
		/// Element access operator
		/// </summary>
		/// <param name="row_id">Index of the row</param>
		/// <param name="col_idj">Index of the column</param>
		/// <returns>Reference to the element with the given row and column indices</returns>
		Real& operator ()(const std::size_t row_id, const std::size_t col_id);

		/// <summary>
		/// Element access operator (constant version)
		/// </summary>
		/// <param name="row_id">Index of the row</param>
		/// <param name="col_idj">Index of the column</param>
		/// <returns>Constant reference to the element with the given row and column indices</returns>
		const Real& operator ()(const std::size_t row_id, const std::size_t col_id) const;

		/// <summary>
		/// Multiplication by a vector from the right
		/// </summary>
		DenseVector friend operator *(const DenseMatrix& matr, const DenseVector& vec);

		/// <summary>
		/// Performs multiplication by the given vector (from the right)
		/// and simultaneously adds another vector to the result of multiplication
		/// </summary>
		/// <param name="mul_vec">The vector that is involved in the multiplication operation</param>
		/// <param name="add_vec">The vector that is involved in the addition operation</param>
		DenseVector mul_add(const DenseVector& mul_vec, const DenseVector& add_vec) const;

		/// <summary>
		/// Multiplication by a vector from the left
		/// </summary>
		DenseVector friend operator *(const DenseVector& vec, const DenseMatrix& matr);

		/// <summary>
		/// Equality operator
		/// </summary>
		bool operator ==(const DenseMatrix& matr) const;

		/// <summary>
		/// Inequality operator
		/// </summary>
		bool operator !=(const DenseMatrix& matr) const;

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
		static inline DenseMatrix random(const std::size_t row_dim, const std::size_t col_dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// Compound addition operator
		/// </summary>
		DenseMatrix& operator +=(const DenseMatrix &mat);

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		DenseMatrix& operator -=(const DenseMatrix &mat);

		/// <summary>
		/// Compound scalar multiplication operator
		/// </summary>
		DenseMatrix& operator *=(const Real& scalar);

		/// <summary>
		/// "Maximal absolute value" norm ("infinity" norm)
		/// </summary>
		Real max_abs() const;

		/// <summary>
		/// Assigns the given value to all the elements of the matrix
		/// </summary>
		void fill(const Real& val);
	};

	/// <summary>
	/// Matrix addition operator
	/// </summary>
	DenseMatrix operator +(const DenseMatrix& mat1, const DenseMatrix& mat2);

	/// <summary>
	/// Matrix subtraction operator
	/// </summary>
	DenseMatrix operator -(const DenseMatrix& mat1, const DenseMatrix& mat2);

	/// <summary>
	/// Matrix by scalar multiplication operator
	/// </summary>
	DenseMatrix operator *(const DenseMatrix& mat, const Real& scalar);

	/// <summary>
	/// Scalar by matrix multiplication operator
	/// </summary>
	DenseMatrix operator *(const Real& scalar, const DenseMatrix& mat);

	/// <summary>
	/// Returns result of multiplication of the given vector-column by the given vector-row
	/// </summary>
	DenseMatrix vector_col_times_vector_row(const DenseVector& vec_col, const DenseVector& vec_row);
}
