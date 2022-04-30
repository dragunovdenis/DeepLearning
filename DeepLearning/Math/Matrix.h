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
#include "BasicCollection.h"

namespace DeepLearning
{
	class Vector;

	/// <summary>
	/// Representation of a dense rectangular matrix
	/// </summary>
	class Matrix : public BasicCollection
	{
		/// <summary>
		/// Elements of the matrix in a "flattened" form
		/// </summary>
		Real* _data{};

		/// <summary>
		/// Matrix dimensions
		/// </summary>
		std::size_t _row_dim{};
		std::size_t _col_dim{};

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

		/// <summary>
		/// Method to free the data array
		/// </summary>
		void free();

	public:

		/// <summary>
		/// Total number of elements in the matrix
		/// </summary>
		std::size_t size() const;

		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const
		{
			const auto proxy = std::vector<Real>(begin(), end());
			msgpack::type::make_define_array(_row_dim, _col_dim, proxy).msgpack_pack(msgpack_pk);
		}

		void msgpack_unpack(msgpack::object const& msgpack_o)
		{
			std::vector<Real> proxy;
			msgpack::type::make_define_array(_row_dim, _col_dim, proxy).msgpack_unpack(msgpack_o);
			_data = reinterpret_cast<Real*>(std::malloc(size() * sizeof(Real)));
			std::copy(proxy.begin(), proxy.end(), begin());
		}

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
		Matrix() = default;

		/// <summary>
		/// Constructs a dense matrix of the given dimensions
		/// </summary>
		Matrix(const std::size_t row_dim, const std::size_t col_dim, const bool assign_zero = true);

		/// <summary>
		/// Constructs a dense matrix of the given dimensions filled according to the given generator function
		/// </summary>
		Matrix(const std::size_t row_dim, const std::size_t col_dim, const std::function<Real()>& generator);

		/// <summary>
		/// Constructs a dense matrix of the given dimensions filled
		/// with a uniformly distributed pseudo-random values from the given range
		/// </summary>
		Matrix(const std::size_t row_dim, const std::size_t col_dim,
			const Real range_begin, const Real range_end);

		/// <summary>
		/// Copy constructor
		/// </summary>
		Matrix(const Matrix& matr);

		/// <summary>
		/// Assignment operator
		/// </summary>
		Matrix& operator =(const Matrix& matr);

		/// <summary>
		/// Move constructor
		/// </summary>
		Matrix(Matrix&& matr) noexcept;

		/// <summary>
		/// Destructor
		/// </summary>
		~Matrix();

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
		Vector friend operator *(const Matrix& matr, const Vector& vec);

		/// <summary>
		/// Performs multiplication by the given vector (from the right)
		/// and simultaneously adds another vector to the result of multiplication
		/// </summary>
		/// <param name="mul_vec">The vector that is involved in the multiplication operation</param>
		/// <param name="add_vec">The vector that is involved in the addition operation</param>
		Vector mul_add(const BasicCollection& mul_vec, const BasicCollection& add_vec) const;

		/// <summary>
		/// Multiplication by a vector from the left
		/// </summary>
		Vector friend operator *(const BasicCollection& vec, const Matrix& matr);

		/// <summary>
		/// Equality operator
		/// </summary>
		bool operator ==(const Matrix& matr) const;

		/// <summary>
		/// Inequality operator
		/// </summary>
		bool operator !=(const Matrix& matr) const;

		/// <summary>
		/// Pointer to the first element of the vector
		/// </summary>
		Real* begin();

		/// <summary>
		/// Pointer to the first element of the vector (constant version)
		/// </summary>
		const Real* begin() const;

		/// <summary>
		/// Pointer to the "behind last" element of the vector
		/// </summary>
		Real* end();

		/// <summary>
		/// Pointer to the "behind last" element of the vector (constant version)
		/// </summary>
		const Real* end() const;

		/// <summary>
		/// Generates a vector filled with uniformly distributed pseudo random values
		/// </summary>
		static inline Matrix random(const std::size_t row_dim, const std::size_t col_dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// Compound addition operator
		/// </summary>
		Matrix& operator +=(const Matrix &mat);

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		Matrix& operator -=(const Matrix &mat);

		/// <summary>
		/// Compound scalar multiplication operator
		/// </summary>
		Matrix& operator *=(const Real& scalar);

		/// <summary>
		/// Method to abandon resources (should be called when the resources are "moved")
		/// </summary>
		void abandon_resources() override;
	};

	/// <summary>
	/// Matrix addition operator
	/// </summary>
	Matrix operator +(const Matrix& mat1, const Matrix& mat2);

	/// <summary>
	/// Matrix subtraction operator
	/// </summary>
	Matrix operator -(const Matrix& mat1, const Matrix& mat2);

	/// <summary>
	/// Matrix by scalar multiplication operator
	/// </summary>
	Matrix operator *(const Matrix& mat, const Real& scalar);

	/// <summary>
	/// Scalar by matrix multiplication operator
	/// </summary>
	Matrix operator *(const Real& scalar, const Matrix& mat);

	/// <summary>
	/// Returns result of multiplication of the given vector-column by the given vector-row
	/// </summary>
	Matrix vector_col_times_vector_row(const BasicCollection& vec_col, const BasicCollection& vec_row);
}
