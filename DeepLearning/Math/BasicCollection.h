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
#include "../Memory/MemHandle.h"
#include "LinAlg3d.h"

namespace DeepLearning
{
	/// <summary>
	/// "Basic" extension that can be useful for a collection-class
	/// </summary>
	class BasicCollection
	{
	protected:
		/// <summary>
		/// Pointer to the elements of collection
		/// </summary>
		Real* _data{};

	public:
		/// <summary>
		/// Reallocates memory of the collection to meet the given number of elements
		/// (if the current "capacity" is lower than the given "new" size)
		/// </summary>
		virtual void resize(const Index3d& size_3d) = 0;

		/// <summary>
		/// Element-wise sum with another collection of the same size
		/// It is a responsibility of the caller to make sure that the collections are of the same size
		/// </summary>
		void add(const BasicCollection& collection);

		/// <summary>
		/// Element-wise sum with another collection of the same size scaled by the given scalar
		/// </summary>
		void add_scaled(const BasicCollection& collection, const Real& scalar);

		/// <summary>
		/// Scales the current collection by "scalar_0" and adds the given collection scaled by "scalar_1"
		/// </summary>
		void scale_and_add_scaled(const Real& scalar_0, const BasicCollection& collection, const Real& scalar_1);

		/// <summary>
		/// Adds the given collection to the current collection scaled by the given factor
		/// </summary>
		void scale_and_add(const BasicCollection& collection, const Real& scalar);

		/// <summary>
		/// Element-wise difference with another collection of the same size
		/// It is a responsibility of the caller to make sure that the collections are of the same size
		/// </summary>
		void sub(const BasicCollection& collection);

		/// <summary>
		/// Multiplies each element of the current collection by the given scalar
		/// </summary>
		void mul(const Real& scalar);

		/// <summary>
		/// Size of the collection
		/// </summary>
		[[nodiscard]] virtual std::size_t size() const = 0;

		/// <summary>
		/// Returns number of preallocated elements ("capacity" can be greater than the size of collection)
		/// </summary>
		[[nodiscard]] virtual std::size_t capacity() const = 0;

		/// <summary>
		/// Pointer to the first element of the vector
		/// </summary>
		Real* begin();

		/// <summary>
		/// Pointer to the first element of the vector (constant version)
		/// </summary>
		[[nodiscard]] const Real* begin() const;

		/// <summary>
		/// Pointer to the "behind last" element of the vector
		/// </summary>
		Real* end();

		/// <summary>
		/// Pointer to the "behind last" element of the vector (constant version)
		/// </summary>
		[[nodiscard]] const Real* end() const;

		/// <summary>
		/// Subscript operator
		/// </summary>
		Real& operator [](const std::size_t& id);

		/// <summary>
		/// Subscript operator (constant version)
		/// </summary>
		const Real& operator [](const std::size_t& id) const;

		/// <summary>
		/// "Maximal absolute value" norm ("infinity" norm) of the collection
		/// </summary>
		[[nodiscard]] Real max_abs() const;

		/// <summary>
		/// Returns sum of all the elements of the collection transformed with the given operator
		/// </summary>
		Real sum(const std::function<Real(Real)>& transform_operator = [](const auto& x) {return x; }) const;

		/// <summary>
		/// Returns sum of squares of the elements in the collection
		/// </summary>
		[[nodiscard]] Real sum_of_squares() const;

		/// <summary>
		/// Assigns the given value to all the elements of the collection
		/// </summary>
		void fill(const Real& val);

		/// <summary>
		/// Returns "true" if the collection is empty
		/// </summary>
		[[nodiscard]] bool empty() const;

		/// <summary>
		/// Performs the Hadamard (element-wise) product operation between the current collection and the input
		/// </summary>
		void hadamard_prod_in_place(const BasicCollection& collection);

		/// <summary>
		/// Performs the Hadamard (element-wise) product operation between the current collection and the input
		/// </summary>
		template<class T>
		T hadamard_prod(const T& input) const
		{
			auto result = input;
			result.hadamard_prod_in_place(*this);
			return result;
		}

		/// <summary>
		/// Calculates dot product with another collection of the same size
		/// </summary>
		[[nodiscard]] Real dot_product(const BasicCollection& collection) const;

		/// <summary>
		/// Returns index of the "maximal element" defined by the given comparer
		/// or "1" if the collection contains zero elements
		/// </summary>
		std::size_t max_element_id(const std::function<bool(Real, Real)>& comparer = [](const auto& a, const auto& b) {return a < b; }) const;

		/// <summary>
		/// Returns maximal element of the collection according to the given comparer or "nan" if the collection is empty
		/// </summary>
		Real max_element(const std::function<bool(Real, Real)>& comparer = [](const auto& a, const auto& b) {return a < b; }) const;

		/// <summary>
		/// Method to abandon resources (should be called when the resources are "moved")
		/// </summary>
		virtual void abandon_resources() = 0;

		/// <summary>
		/// Converter to std::vector
		/// </summary>
		/// <returns></returns>
		[[nodiscard]] std::vector<Real> to_stdvector() const;

		/// <summary>
		/// Returns read-only memory handle to the data array of the collection
		/// </summary>
		[[nodiscard]] RealMemHandleConst get_handle() const;

		/// <summary>
		/// Returns memory handle to the data array of the collection
		/// </summary>
		RealMemHandle get_handle();

		/// <summary>
		/// Fills the collection with normally distributed random values with zero mean
		/// and the given standard deviation "sigma", which is by default equal to one divided
		/// by the square root of the elements in the collection
 		/// </summary>
		void standard_random_fill(const Real& sigma = -1);

		/// <summary>
		/// Returns "true" if at least one element of the collection is "nan"
		/// </summary>
		[[nodiscard]] bool is_nan() const;

		/// <summary>
		/// Returns "true" if at least one element of the collection is infinite (positive or negative)
		/// </summary>
		[[nodiscard]] bool is_inf() const;

		/// <summary>
		/// Virtual destructor to ensure that resources of the descending classes are properly released
		/// </summary>
		virtual ~BasicCollection() = default;
	};
}
