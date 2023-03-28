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
#include <msgpack.hpp>
#include <vector>
#include "../Utilities.h"

namespace DeepLearning
{
	/// <summary>
	/// N-dimensional vector on stack
	/// </summary>
	template <class T, int N>
	class VectorNd
	{
		T _data[N] {};//the underlying data

		/// <summary>
		/// Returns "true" if the given integer value represents a valid index to access elements of the vector
		/// </summary>
		static bool check_bounds(const int i)
		{
			return i >= 0 && i < N;
		}

	public:
		MSGPACK_DEFINE(_data);

		/// <summary>
		/// Returns randomly assigned instance of the vector
		/// </summary>
		static VectorNd random(const T& min = T(0), const T& max = T(1))
		{
			VectorNd result;
			Utils::fill_with_random_values(result._data, result._data + N, min, max);

			return result;
		}

		/// <summary>
		/// Compound addition operator
		/// </summary>
		VectorNd& operator +=(const VectorNd& vec)
		{
			for (auto i = 0; i < N; ++i)
				_data[i] += vec._data[i];

			return *this;
		}

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		VectorNd& operator -=(const VectorNd& vec)
		{
			for (auto i = 0; i < N; ++i)
				_data[i] -= vec._data[i];

			return *this;
		}

		/// <summary>
		/// Compound multiplication operator
		/// </summary>
		VectorNd& operator *=(const T& scalar)
		{
			for (auto i = 0; i < N; ++i)
				_data[i] *= scalar;

			return *this;
		}

		/// <summary>
		/// Returns L-infinity norm of the vector
		/// </summary>
		T max_abs() const
		{
			T result = T(0);
			for (auto i = 0; i < N; ++i)
				result = std::max(result, std::abs(_data[i]));

			return result;
		}

		/// <summary>
		/// Returns Euclidean norm of the vector
		/// </summary>
		T length() const
		{
			T sum_of_squares = T(0);
			for (auto i = 0; i < N; ++i)
				sum_of_squares = _data[i] * _data[i];

			return std::sqrt(sum_of_squares);
		}

		/// <summary>
		/// Equality operator
		/// </summary>
		bool operator ==(const VectorNd& vec) const
		{
			for (auto i = 0; i < N; ++i)
				if (_data[i] != vec._data[i])
					return false;

			return true;
		}

		/// <summary>
		/// Inequality operator
		/// </summary>
		bool operator !=(const VectorNd& vec) const
		{
			return !(*this == vec);
		}

		/// <summary>
		/// Negation operator
		/// </summary>
		VectorNd operator -() const
		{
			VectorNd result;

			for (auto i = 0; i < N; ++i)
				result._data[i] = -_data[i];

			return result;
		}

		/// <summary>
		/// Sub-script operator
		/// </summary>
		T& operator [](const int i)
		{
			#ifdef CHECK_BOUNDS
			if (!check_bounds(i))
				throw std::exception("Index out of bounds");
			#endif
			return _data[i];
		}

		/// <summary>
		/// Sub-script operator (constant version)
		/// </summary>
		const T& operator [](const int i) const
		{
			#ifdef CHECK_BOUNDS
			if (!check_bounds(i))
				throw std::exception("Index out of bounds");
			#endif
			return _data[i];
		}

		/// <summary>
		/// Conversion to "standard" vector
		/// </summary>
		[[nodiscard]] std::vector<T> to_std_vector() const
		{
			std::vector<T> result;
			result.assign(_data, _data + N);

			return result;
		}

		/// <summary>
		/// Assignment from "standard" vector
		/// </summary>
		VectorNd& assign(const std::vector<T>& vec)
		{
			if (vec.size() != N)
				throw std::exception("Size mismatch");

			std::copy(vec.begin(), vec.end(), _data);

			return *this;
		}

		/// <summary>
		///	Returns a human-readable representation of the vector
		/// </summary>
		[[nodiscard]] std::string to_string() const
		{
			return Utils::vector_to_str(to_std_vector());
		}

		/// <summary>
		/// Tries to assign vector from the given string. Returns "true" if succeeded
		/// </summary>
		static bool try_parse(std::string& str, VectorNd& out)
		{
			try
			{
				out.assign(Utils::parse_vector<T>(str));
			} catch (...)
			{
				return false;
			}

			return true;
		}

		/// <summary>
		/// Returns index of the first minimal element
		/// </summary>
		[[nodiscard]] int min_elem_id() const
		{
			T min_val = std::numeric_limits<T>::max();
			int result = -1;

			for (int i = 0; i< N; ++i)
			{
				if (_data[i] < min_val)
				{
					min_val = _data[i];
					result = i;
				}
			}

			return result;
		}
	};

	/// <summary>
	/// Addition operator
	/// </summary>
	template <class T, int N>
	VectorNd<T, N> operator +(const VectorNd<T, N>& vec1, const VectorNd<T, N>& vec2)
	{
		auto result = vec1;
		return result += vec2;
	}

	/// <summary>
	/// Subtraction operator
	/// </summary>
	template <class T, int N>
	VectorNd<T, N> operator -(const VectorNd<T, N>& vec1, const VectorNd<T, N>& vec2)
	{
		auto result = vec1;
		return result -= vec2;
	}

	/// <summary>
	/// Multiplication by scalar operation
	/// </summary>
	template <class T, int N>
	VectorNd<T, N> operator *(const VectorNd<T, N>& vec, const T& scalar)
	{
		auto result = vec;
		return result *= scalar;
	}

	/// <summary>
	/// Multiplication by scalar operation
	/// </summary>
	template <class T, int N>
	VectorNd<T, N> operator *(const T& scalar, const VectorNd<T, N>& vec)
	{
		return vec * scalar;
	}
}
