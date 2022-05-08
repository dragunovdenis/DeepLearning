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


#include "../Utilities.h"
#include <msgpack.hpp>

#pragma once

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
		Vector3d<T>& operator +=(const Vector3d<T>& vec)
		{
			x += vec.x;
			y += vec.y;
			z += vec.z;

			return *this;
		}

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		Vector3d<T>& operator -=(const Vector3d<T>& vec)
		{
			x -= vec.x;
			y -= vec.y;
			z -= vec.z;

			return *this;
		}

		/// <summary>
		/// Compound multiplication operator
		/// </summary>
		Vector3d<T>& operator *=(const T& scalar)
		{
			x *= scalar;
			y *= scalar;
			z *= scalar;

			return *this;
		}

		/// <summary>
		/// Returns random vector
		/// </summary>
		static Vector3d<T> random(const T& min = -1, const T& max = 1)
		{
			return { T(Utils::get_random(min, max)), T(Utils::get_random(min, max)), T(Utils::get_random(min, max)) };
		}

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
		Vector3d<T> operator -() const
		{
			return { -x, -y, -z };
		}

		/// <summary>
		/// Hadamard (component-wise) product of the vectors
		/// </summary>
		Vector3d<T> hadamard_prod(const Vector3d<T>& vec) const
		{
			return { x * vec.x, y * vec.y, z * vec.z };
		}

		/// <summary>
		/// Default constructor
		/// </summary>
		Vector3d() {}

		/// <summary>
		/// Constructor by 3 coordinates
		/// </summary>
		Vector3d(const T& x_, const T& y_, const T& z_):x(x_), y(y_), z(z_) {}

		/// <summary>
		/// Constructors a vector with all the coordinates equal to the given value `w`
		/// </summary>
		Vector3d(const T& w): x(w), y(w), z(w){}
	};

	/// <summary>
	/// Addition operator
	/// </summary>
	template <class T>
	Vector3d<T> operator +(const Vector3d<T>& vec1, const Vector3d<T>& vec2)
	{
		auto result = vec1;
		return result += vec2;
	}

	/// <summary>
	/// Subtraction operator
	/// </summary>
	template <class T>
	Vector3d<T> operator -(const Vector3d<T>& vec1, const Vector3d<T>& vec2)
	{
		auto result = vec1;
		return result -= vec2;
	}

	template <class T>
	Vector3d<T> operator *(const Vector3d<T>& vec, const T& scalar)
	{
		auto result = vec;
		return result *= scalar;
	}

	template <class T>
	Vector3d<T> operator *(const T& scalar, const Vector3d<T>& vec)
	{
		return vec * scalar;
	}

	using Index3d = Vector3d<long long>;
}
