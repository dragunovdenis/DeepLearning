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
#include <filesystem>
#include "BasicCudaCollection.cuh"
#include "Matrix.h"
#include "LinAlg3d.h"

namespace DeepLearning
{
	class CudaVector;

	/// <summary>
	/// Represents a dense vector of arbitrary dimension
	/// </summary>
	class CudaMatrix : public BasicCudaCollection
	{
		/// <summary>
		/// Matrix dimensions
		/// </summary>
		std::size_t _row_dim{};
		std::size_t _col_dim{};

		/// <summary>
		/// Method to free the data array
		/// </summary>
		void free();

		/// <summary>
		/// Assignment from another CUDA matrix
		/// </summary>
		void assign(const CudaMatrix& source);

		/// <summary>
		/// Assignment from a "host" matrix
		/// </summary>
		void assign(const Matrix& source);

		/// <summary>
		/// Reallocates memory of the matrix to meet the given number of elements
		/// (if the current size does not coincide with the given "new" size)
		/// </summary>
		void resize(const std::size_t& new_row_dim, const std::size_t& new_col_dim);

	public:

		using Base = BasicCudaCollection;

		/// <summary>
		/// Total number of elements in the matrix
		/// </summary>
		std::size_t size() const;

		/// <summary>
		/// Returns size of the collection in a "unified" form
		/// </summary>
		Index3d size_3d() const;

		/// <summary>
		/// Converts the current instance of CUDA vector into its "host" counterpart
		/// </summary>
		Matrix to_host() const;

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const
		{
			const auto proxy = to_host();
			msgpack::type::make_define_array(proxy).msgpack_pack(msgpack_pk);
		}

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

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
		CudaMatrix() = default;

		/// <summary>
		/// Constructs a dense matrix of the given dimensions
		/// </summary>
		CudaMatrix(const std::size_t row_dim, const std::size_t col_dim, const bool assign_zero = true);

		/// <summary>
		/// Constructs a dense matrix of the given dimensions ("unified" form)
		/// </summary>
		CudaMatrix(const Index3d& size, const bool assign_zero = true);

		/// <summary>
		/// Constructs a dense matrix of the given dimensions filled
		/// with a uniformly distributed pseudo-random values from the given range
		/// </summary>
		CudaMatrix(const std::size_t row_dim, const std::size_t col_dim,
			const Real range_begin, const Real range_end);

		/// <summary>
		/// Copy constructor
		/// </summary>
		CudaMatrix(const CudaMatrix& matr);

		/// <summary>
		/// Assignment operator
		/// </summary>
		CudaMatrix& operator =(const CudaMatrix& matr);

		/// <summary>
		/// Move constructor
		/// </summary>
		CudaMatrix(CudaMatrix&& matr) noexcept;

		/// <summary>
		/// Destructor
		/// </summary>
		~CudaMatrix();

		/// <summary>
		/// Multiplication by a vector from the right
		/// </summary>
		CudaVector friend operator *(const CudaMatrix& matr, const CudaVector & vec);

		/// <summary>
		/// Performs multiplication by the given vector (from the right)
		/// and simultaneously adds another vector to the result of multiplication
		/// </summary>
		/// <param name="mul_vec">The vector that is involved in the multiplication operation</param>
		/// <param name="add_vec">The vector that is involved in the addition operation</param>
		CudaVector mul_add(const BasicCudaCollection & mul_vec, const BasicCudaCollection & add_vec) const;

		/// <summary>
		/// Multiplication by a vector from the left
		/// </summary>
		CudaVector friend operator *(const BasicCudaCollection & vec, const CudaMatrix& matr);

		/// <summary>
		/// Equality operator
		/// </summary>
		bool operator ==(const CudaMatrix& matr) const;

		/// <summary>
		/// Inequality operator
		/// </summary>
		bool operator !=(const CudaMatrix & matr) const;

		/// <summary>
		/// Generates a vector filled with uniformly distributed pseudo random values
		/// </summary>
		static CudaMatrix random(const std::size_t row_dim, const std::size_t col_dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// Compound addition operator
		/// </summary>
		CudaMatrix& operator +=(const CudaMatrix& mat);

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		CudaMatrix& operator -=(const CudaMatrix& mat);

		/// <summary>
		/// Compound scalar multiplication operator
		/// </summary>
		CudaMatrix& operator *=(const Real & scalar);

		/// <summary>
		/// Method to abandon resources (should be called when the resources are "moved")
		/// </summary>
		void abandon_resources() override;

		/// <summary>
		/// Logs the matrix to a text file
		/// </summary>
		/// <param name="filename">Full name of the log file on disk</param>
		void log(const std::filesystem::path & file_name) const;
	};

	/// <summary>
	/// Matrix addition operator
	/// </summary>
	CudaMatrix operator +(const CudaMatrix& mat1, const CudaMatrix& mat2);

	/// <summary>
	/// Matrix subtraction operator
	/// </summary>
	CudaMatrix operator -(const CudaMatrix& mat1, const CudaMatrix& mat2);

	/// <summary>
	/// Matrix by scalar multiplication operator
	/// </summary>
	CudaMatrix operator *(const CudaMatrix& mat, const Real& scalar);

	/// <summary>
	/// Scalar by matrix multiplication operator
	/// </summary>
	CudaMatrix operator *(const Real& scalar, const CudaMatrix& mat);

	/// <summary>
	/// Returns result of multiplication of the given vector-column by the given vector-row
	/// </summary>
	CudaMatrix vector_col_times_vector_row(const BasicCudaCollection& vec_col, const BasicCudaCollection& vec_row);
}