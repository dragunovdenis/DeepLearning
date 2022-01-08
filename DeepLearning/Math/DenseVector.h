#pragma once

#include <vector>
#include "../defs.h"
#include <msgpack.hpp>

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
	};
}
