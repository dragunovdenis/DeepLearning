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
#include "BasicCollection.h"
#include <filesystem>
#include "LinAlg3d.h"

namespace DeepLearning
{
	/// <summary>
	/// Represents a dense vector of arbitrary dimension
	/// </summary>
	class Vector : public BasicCollection
	{
		/// <summary>
		/// Number of elements in the vector
		/// </summary>
		std::size_t _dim{};

		/// <summary>
		/// Number of reserved elements
		/// </summary>
		std::size_t _capacity{};

		/// <summary>
		/// Returns "true" if the given index is valid (within the boundaries of the underlying "data" collection)
		/// </summary>
		bool check_bounds(const ::std::size_t id) const;

		/// <summary>
		/// Frees the allocated memory
		/// </summary>
		void free();

		/// <summary>
		/// Assignment from another Vector
		/// </summary>
		template <class S>
		void assign(const S& source);

	public:

		using Base = BasicCollection;

		/// <summary>
		/// Reallocates memory of the tensor to meet the given number of elements
		/// (if the current "capacity" is lower than the given "new" size)
		/// </summary>
		void resize(const std::size_t& new_size);

		/// <summary>
		/// Reallocates memory of the tensor to meet the given number of elements
		/// (if the current "capacity" is lower than the given "new" size)
		/// </summary>
		void resize(const Index3d& size_3d) override;

		/// <summary>
		/// Size of the vector
		/// </summary>
		std::size_t size() const override;

		/// <summary>
		/// Returns number of allocated (reserved) elements (can be greater or equal to size)
		/// </summary>
		std::size_t capacity() const override;

		/// <summary>
		/// Returns size of the collection in a "unified" form
		/// </summary>
		Index3d size_3d() const;

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer> 
		void msgpack_pack(Packer& msgpack_pk) const 
		{ 
			const auto proxy = to_stdvector();
			msgpack::type::make_define_array(proxy).msgpack_pack(msgpack_pk);
		} 

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

		/// <summary>
		/// Default constructor
		/// </summary>
		Vector() = default;

		/// <summary>
		/// Copy constructor
		/// </summary>
		Vector(const Vector& vec);

		/// <summary>
		/// Move constructor
		/// </summary>
		Vector(Vector&& vec) noexcept;

		/// <summary>
		/// Copy assignment operator
		/// </summary>
		Vector& operator=(const Vector& vec);

		/// <summary>
		/// Constructs dense vector of the given dimension
		/// </summary>
		Vector(const std::size_t dim, const bool assign_zero = true);

		/// <summary>
		/// Constructs dense vector of the given dimension (unified version)
		/// </summary>
		Vector(const Index3d& size, const bool assign_zero = true);

		/// <summary>
		/// Constructor from given source vector
		/// </summary>
		template <class T>
		Vector(const std::vector<T>& source);

		/// <summary>
		/// Constructs dense vector of the given dimension filled with
		/// uniformly distributed pseudo-random values from the given range
		/// </summary>
		Vector(const std::size_t dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// Destructor
		/// </summary>
		~Vector();

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
		Vector& operator += (const Vector& vec);

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		Vector& operator -= (const Vector& vec);

		/// <summary>
		/// Compound scalar multiplication operator
		/// </summary>
		Vector& operator *= (const Real& scalar);

		/// <summary>
		/// Equality operator
		/// </summary>
		bool operator ==(const Vector& vect) const;

		/// <summary>
		/// Inequality operator
		/// </summary>
		bool operator !=(const Vector& vect) const;

		/// <summary>
		/// Generates a vector filled with uniformly distributed pseudo random values
		/// </summary>
		static Vector random(const std::size_t dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// Method to abandon resources (should be called when the resources are "moved")
		/// </summary>
		void abandon_resources() override;

		/// <summary>
		/// Logs the vector to a text file
		/// </summary>
		/// <param name="filename">Full name of the log file on disk</param>
		void log(const std::filesystem::path& filename) const;
	};

	/// <summary>
	/// Vector addition operator
	/// </summary>
	Vector operator + (const Vector& vec1, const Vector& vec2);

	/// <summary>
	/// Vector subtraction operator
	/// </summary>
	Vector operator -(const Vector& vec1, const Vector& vec2);

	/// <summary>
	/// Vector-Scalar multiplication operator
	/// </summary>
	Vector operator *(const Vector& vec, const Real& scalar);

	/// <summary>
	/// Scalar-vector multiplication operator
	/// </summary>
	Vector operator *(const Real& scalar, const Vector& vec);
}
