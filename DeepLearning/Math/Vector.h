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
		/// Returns "true" if the given index is valid (within the boundaries of the underlying "data" collection)
		/// </summary>
		bool check_bounds(const ::std::size_t id) const;

		/// <summary>
		/// Assignment from another Vector
		/// </summary>
		template <class S>
		void assign(const S& source);

	protected:
		/// <summary>
		/// Method to abandon resources (should be called when the resources are being "moved")
		/// </summary>
		void abandon_resources() override;

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
		/// Returns size of the collection in a "unified" form
		/// </summary>
		Index3d size_3d() const override;

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
		/// Move assignment operator
		/// </summary>
		Vector& operator=(Vector&& vec) noexcept;

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
		Vector(const std::size_t dim, const Real range_begin, const Real range_end, std::mt19937* seeder = nullptr);

		/// <summary>
		/// Returns dimension of the vector
		/// </summary>
		std::size_t dim() const;

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
		bool operator ==(const Vector& vec) const;

		/// <summary>
		/// Inequality operator
		/// </summary>
		bool operator !=(const Vector& vec) const;

		/// <summary>
		/// Logs the vector to a text file
		/// </summary>
		/// <param name="filename">Full name of the log file on disk</param>
		void log(const std::filesystem::path& filename) const;

		/// <summary>
		/// Fills the current vector randomly with 1s and 0s, so that eventually it contains exactly
		/// 'selected_cnt' elements initialized with `1`; If `selected_cnt` is greater or equal to
		/// the size of the vector then after the call all the elements of the vector are initialized with `1`
		/// </summary>
		/// <param name="selected_cnt">Number of 1s in the collection after the call</param>
		/// <param name="aux_collection">An auxiliary collection, can be allocated on the caller side and
		/// provides a possibility to avoid unnecessary re-allocations when the method below is called multiple times</param>
		/// <param name="seeder">Seed value provider (for pseudo-random generators)</param>
		void fill_with_random_selection_map(const std::size_t& selected_cnt,
			std::vector<int>& aux_collection, std::mt19937* seeder = nullptr);

		/// <summary>
		/// Less optimal but more simple version of the method above
		/// </summary>
		/// <param name="selected_cnt">Number of selected elements from the current collection</param>
		void fill_with_random_selection_map(const std::size_t& selected_cnt);
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
