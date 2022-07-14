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
#include "Vector.h"
#include <filesystem>
#include "BasicCudaCollection.cuh"
#include "LinAlg3d.h"

namespace DeepLearning
{
	/// <summary>
	/// Represents a dense vector of arbitrary dimension
	/// </summary>
	class CudaVector : public BasicCudaCollection
	{
		/// <summary>
		/// Number of elements in the vector
		/// </summary>
		std::size_t _dim{};

		/// <summary>
		/// Frees the allocated memory
		/// </summary>
		void free();

		/// <summary>
		/// Assignment from another CUDA collection
		/// </summary>
		void assign(const BasicCudaCollection& source);

		/// <summary>
		/// Assignment from a "host" collection
		/// </summary>
		void assign(const BasicCollection& source);

		/// <summary>
		/// Reallocates memory of the vector to meet the given number of elements
		/// (if the current size does not coincide with the given "new" size)
		/// </summary>
		void resize(const std::size_t& new_size);

	public:

		using Base = BasicCudaCollection;

		/// <summary>
		/// Size of the vector
		/// </summary>
		std::size_t size() const;

		/// <summary>
		/// Returns size of the collection in a "unified" form
		/// </summary>
		Index3d size_3d() const;

		/// <summary>
		/// Converts the current instance of CUDA vector into its "host" counterpart
		/// </summary>
		Vector to_host() const;

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
		/// Default constructor
		/// </summary>
		CudaVector() = default;

		/// <summary>
		/// Copy constructor
		/// </summary>
		CudaVector(const CudaVector& vec);

		/// <summary>
		/// Move constructor
		/// </summary>
		CudaVector(CudaVector&& vec) noexcept;

		/// <summary>
		/// Copy assignment operator
		/// </summary>
		CudaVector& operator=(const CudaVector& vec);

		/// <summary>
		/// Constructs dense vector of the given dimension
		/// </summary>
		CudaVector(const std::size_t dim, const bool assign_zero = true);

		/// <summary>
		/// Constructs dense vector of the given dimension ("unified" version)
		/// </summary>
		CudaVector(const Index3d& size, const bool assign_zero = true);

		/// <summary>
		/// Constructs dense vector of the given dimension filled with
		/// uniformly distributed pseudo-random values from the given range
		/// </summary>
		CudaVector(const std::size_t dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// Destructor
		/// </summary>
		~CudaVector();

		/// <summary>
		/// Returns dimension of the vector
		/// </summary>
		std::size_t dim() const;

		/// <summary>
		/// Compound addition operator
		/// </summary>
		CudaVector& operator += (const CudaVector& vec);

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		CudaVector& operator -= (const CudaVector& vec);

		/// <summary>
		/// Compound scalar multiplication operator
		/// </summary>
		CudaVector& operator *= (const Real& scalar);

		/// <summary>
		/// Equality operator
		/// </summary>
		bool operator ==(const CudaVector& vect) const;

		/// <summary>
		/// Inequality operator
		/// </summary>
		bool operator !=(const CudaVector& vect) const;

		/// <summary>
		/// Generates a vector filled with uniformly distributed pseudo random values
		/// </summary>
		static CudaVector random(const std::size_t dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// Method to abandon resources (should be called when the resources are "moved")
		/// </summary>
		void abandon_resources() override;

		/// <summary>
		/// Logs the vector to a text file
		/// </summary>
		/// <param name="filename">Full name of the log file on disk</param>
		void log(const std::filesystem::path& file_name) const;
	};

	/// <summary>
	/// Vector addition operator
	/// </summary>
	CudaVector operator + (const CudaVector& vec1, const CudaVector& vec2);

	/// <summary>
	/// Vector subtraction operator
	/// </summary>
	CudaVector operator -(const CudaVector& vec1, const CudaVector& vec2);

	/// <summary>
	/// Vector-Scalar multiplication operator
	/// </summary>
	CudaVector operator *(const CudaVector& vec, const Real& scalar);

	/// <summary>
	/// Scalar-vector multiplication operator
	/// </summary>
	CudaVector operator *(const Real& scalar, const CudaVector& vec);

}