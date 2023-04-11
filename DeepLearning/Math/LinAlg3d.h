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
#include "LinAlg2d.h"
#include "../CudaBridge.h"

namespace DeepLearning
{
	/// <summary>
	/// 3-dimensional vector
	/// </summary>
	template <class T>
	struct Vector3d
	{
		/// <summary>
		/// "x" coordinate
		/// </summary>
		T x{};

		/// <summary>
		/// "y" coordinate
		/// </summary>
		T y{};

		/// <summary>
		/// "z" coordinate
		/// </summary>
		T z{};

		MSGPACK_DEFINE(x, y, z);

		/// <summary>
		/// Compound addition operator
		/// </summary>
		CUDA_CALLABLE Vector3d<T>& operator +=(const Vector3d<T>& vec)
		{
			x += vec.x;
			y += vec.y;
			z += vec.z;

			return *this;
		}

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		CUDA_CALLABLE Vector3d<T>& operator -=(const Vector3d<T>& vec)
		{
			x -= vec.x;
			y -= vec.y;
			z -= vec.z;

			return *this;
		}

		/// <summary>
		/// Compound multiplication operator
		/// </summary>
		CUDA_CALLABLE Vector3d<T>& operator *=(const T& scalar)
		{
			x *= scalar;
			y *= scalar;
			z *= scalar;

			return *this;
		}

		/// <summary>
		/// Returns random vector
		/// </summary>
		static Vector3d<T> random(const T& min = -1, const T& max = 1);

		/// <summary>
		/// Returns L-infinity norm of the vector 
		/// </summary>
		T max_abs() const
		{
			return std::max(std::abs(x), std::max(std::abs(y), std::abs(z)));
		}

		/// <summary>
		/// Equality operator
		/// </summary>
		bool operator ==(const Vector3d<T>& vec) const
		{
			return x == vec.x && y == vec.y && z == vec.z;
		}

		/// <summary>
		/// Not equality operator
		/// </summary>
		bool operator !=(const Vector3d<T>& vec) const
		{
			return !(*this == vec);
		}

		/// <summary>
		/// Negation operator
		/// </summary>
		CUDA_CALLABLE Vector3d<T> operator -() const
		{
			return { -x, -y, -z };
		}

		/// <summary>
		/// Hadamard (component-wise) product of the vectors
		/// </summary>
		CUDA_CALLABLE [[nodiscard]] Vector3d<T> hadamard_prod(const Vector3d<T>& vec) const
		{
			return { x * vec.x, y * vec.y, z * vec.z };
		}

		/// <summary>
		/// Default constructor
		/// </summary>
		Vector3d() = default;

		/// <summary>
		/// Constructor by 3 coordinates
		/// </summary>
		CUDA_CALLABLE Vector3d(const T& x_, const T& y_, const T& z_):x(x_), y(y_), z(z_) {}

		/// <summary>
		/// Constructors a vector with all the coordinates equal to the given value `w`
		/// </summary>
		CUDA_CALLABLE Vector3d(const T& w): x(w), y(w), z(w){}

		/// <summary>
		/// Creates vector with the given base type out of the three given
		/// coordinate values (that can be of different types)
		/// </summary>
		template <class P>
		CUDA_CALLABLE Vector3d(const P& x_, const P& y_, const P& z_) : x(static_cast<T>(x_)), y(static_cast<T>(y_)), z(static_cast<T>(z_))
		{}

		/// <summary>
		/// Returns product of coordinates
		/// </summary>
		CUDA_CALLABLE T coord_prod() const
		{
			return x * y * z;
		}

		/// <summary>
		/// Returns a human-readable representation of the vector
		/// </summary>
		[[nodiscard]] std::string to_string() const;

		/// <summary>
		/// Tries to parse the given string consisting of 3 comma, semicolon or space separated sub-strings that are "compatible" with type `R`
		/// into an instance of Vector3d<R>. Returns "true" if parsing succeeds, in which case "out" argument is assigned with parsed values.
		/// Otherwise, "out" argument is assumed to be invalid
		/// </summary>
		static bool try_parse(const std::string& str, Vector3d<T>& out);

		/// <summary>
		/// Returns XY projection of the vector
		/// </summary>
		[[nodiscard]] Vector2d<T> xy() const
		{
			return { x, y };
		}

		/// <summary>
		/// Returns XZ projection of the vector
		/// </summary>
		[[nodiscard]] Vector2d<T> xz() const
		{
			return { x, z };
		}

		/// <summary>
		/// Returns YZ projection of the vector
		/// </summary>
		[[nodiscard]] Vector2d<T> yz() const
		{
			return { y, z };
		}
	};

	/// <summary>
	/// Addition operator
	/// </summary>
	template <class T>
	CUDA_CALLABLE Vector3d<T> operator +(const Vector3d<T>& vec1, const Vector3d<T>& vec2)
	{
		auto result = vec1;
		return result += vec2;
	}

	/// <summary>
	/// Subtraction operator
	/// </summary>
	template <class T>
	CUDA_CALLABLE Vector3d<T> operator -(const Vector3d<T>& vec1, const Vector3d<T>& vec2)
	{
		auto result = vec1;
		return result -= vec2;
	}

	/// <summary>
	/// Multiplication by scalar operator
	/// </summary>
	template <class T>
	CUDA_CALLABLE Vector3d<T> operator *(const Vector3d<T>& vec, const T& scalar)
	{
		auto result = vec;
		return result *= scalar;
	}

	/// <summary>
	/// Multiplication by scalar operator
	/// </summary>
	template <class T>
	CUDA_CALLABLE Vector3d<T> operator *(const T& scalar, const Vector3d<T>& vec)
	{
		return vec * scalar;
	}

	using Index3d = Vector3d<long long>;

	/// <summary>
	/// Linear rectifier function of integer argument
	/// </summary>
	inline CUDA_CALLABLE  long long relu(const long long x)
	{
		return x > 0ll ? x : 0ll;
	}

	/// <summary>
	/// Linear rectifier function, 3d version
	/// </summary>
	inline CUDA_CALLABLE  Index3d relu(const Index3d& v)
	{
		return { relu(v.x), relu(v.y), relu(v.z) };
	}
}
