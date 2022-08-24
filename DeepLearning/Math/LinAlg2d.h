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

#include "../Utilities.h"
#include "../defs.h"
#include <exception>

namespace DeepLearning
{
	/// <summary>
	/// 2-dimensional vector
	/// </summary>
	template <class R = Real>
	struct Vector2d
	{
		/// <summary>
		/// X coordinate
		/// </summary>
		R x{};

		/// <summary>
		/// Y coordinate
		/// </summary>
		R y{};

		/// <summary>
		/// Returns the OX (basis) unity vector
		/// </summary>
		static Vector2d<R> OX()
		{
			return { Real(1), Real(0) };
		}

		/// <summary>
		/// Returns the OY (basis) unity vector
		/// </summary>
		static Vector2d<R> OY()
		{
			return { Real(0), Real(1) };
		}

		/// <summary>
		/// Returns zero vector
		/// </summary>
		static Vector2d<R> zero()
		{
			return { R(0), R(0) };
		}

		/// <summary>
		/// Returns random vector
		/// </summary>
		static Vector2d<R> random()
		{
			return { R(Utils::get_random(-1, 1)), R(Utils::get_random(-1, 1)) };
		}

		/// <summary>
		/// Compound addition operator
		/// </summary>
		Vector2d& operator +=(const Vector2d& vec)
		{
			x += vec.x;
			y += vec.y;

			return *this;
		}

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		Vector2d& operator -=(const Vector2d& vec)
		{
			x -= vec.x;
			y -= vec.y;

			return *this;
		}

		/// <summary>
		/// Compound multiplication by scalar operator
		/// </summary>
		Vector2d& operator *=(const R& scalar)
		{
			x *= scalar;
			y *= scalar;

			return *this;
		}

		/// <summary>
		/// Division by scalar operator
		/// </summary>
		Vector2d& operator /=(const R& scalar)
		{
			const auto scalar_inverse = R(1) / scalar;
			(*this) *= (scalar_inverse);

			return *this;
		}

		/// <summary>
		/// Dot product with the given vector
		/// </summary>
		R dot(const Vector2d<R>& vec) const
		{
			return x * vec.x + y * vec.y;
		}

		/// <summary>
		/// Returns squared Euclidean norm of the vector
		/// </summary>
		/// <returns></returns>
		R norm_sqr() const
		{
			return dot(*this);
		}

		/// <summary>
		/// Returns Euclidean norm of the vector
		/// </summary>
		R norm() const
		{
			return std::sqrt(norm_sqr());
		}

		/// <summary>
		/// L-infinity norm of the vector
		/// </summary>
		R max_abs() const
		{
			return std::max<R>(std::abs(x), std::abs(y));
		}

		/// <summary>
		/// Returns normalized vector
		/// In case the current vector is "zero" the result of normalization will be NaN
		/// </summary>
		Vector2d<R> normalize() const
		{
			return *this / norm();
		}

		/// <summary>
		/// Default constructor
		/// </summary>
		Vector2d() = default;

		/// <summary>
		/// Constructor by 2 coordinates
		/// </summary>
		Vector2d(const R& x_, const R& y_) :x(x_), y(y_) {}

		/// <summary>
		/// Constructors a vector with all the coordinates equal to the given value `w`
		/// </summary>
		Vector2d(const R& w) : x(w), y(w) {}

		/// <summary>
		/// Returns a human-readable representation of the vector
		/// </summary>
		std::string to_string() const
		{
			return std::string("{") + Utils::to_string(x) + ", " + Utils::to_string(y) + "}";
		}

		/// <summary>
		/// Tries to parse the given string consisting of 2 comma, semicolon or space separated sub-strings that are "compatible" with type `R`
		/// into an instance of Vector2d<R>. Returns "true" if parsing succeeds, in which case "out" argument is assigned with parsed values.
		/// Otherwise, "out" argument is assumed to be invalid
		/// </summary>
		static bool try_parse(const std::string& str, Vector2d<R>& out)
		{
			const auto scalars = Utils::parse_scalars<R>(str);

			if (scalars.size() == 1)
			{
				out.x = scalars[0];
				out.y = scalars[0];
				return true;
			}

			if (scalars.size() == 2)
			{
				out.x = scalars[0];
				out.y = scalars[1];
				return true;
			}

			return false;
		}
	};

	/// <summary>
	/// Vector addition operator
	/// </summary>
	template <class R>
	Vector2d<R> operator +(const Vector2d<R>& vec1, const Vector2d<R>& vec2)
	{
		auto result = vec1;
		return result += vec2;
	}

	/// <summary>
	/// Vector subtraction operator
	/// </summary>
	template <class R>
	Vector2d<R> operator -(const Vector2d<R>& vec1, const Vector2d<R>& vec2)
	{
		auto result = vec1;
		return result -= vec2;
	}

	/// <summary>
	/// Multiplication by scalar from the right
	/// </summary>
	template <class R>
	Vector2d<R> operator *(const Vector2d<R>& vec, const R& scalar)
	{
		auto result = vec;
		return result *= scalar;
	}

	/// <summary>
	/// Multiplication by scalar from the left
	/// </summary>
	template <class R>
	Vector2d<R> operator *(const R& scalar, const Vector2d<R>& vec)
	{
		return  vec * scalar;
	}

	/// <summary>
	/// Division by scalar
	/// </summary>
	template <class R>
	Vector2d<R> operator /(const Vector2d<R>& vec, const R& scalar)
	{
		auto result = vec;
		return result /= scalar;
	}

	/// <summary>
	/// Equality operator
	/// </summary>
	template <class R>
	bool operator ==(const Vector2d<R>& vec1, const Vector2d<R>& vec2)
	{
		return vec1.x == vec2.x && vec1.y == vec2.y;
	}

	/// <summary>
	/// "Not equal to" operator
	/// </summary>
	template <class R>
	bool operator !=(const Vector2d<R>& vec1, const Vector2d<R>& vec2)
	{
		return !(vec1 == vec2);
	}

	/// <summary>
	/// Opposite sign operator
	/// </summary>
	template <class R>
	Vector2d<R> operator -(const Vector2d<R>& vec)
	{
		return { -vec.x, -vec.y };
	}

	/// <summary>
	/// 2x2 matrix
	/// </summary>
	template <class R = Real>
	struct Matrix2x2
	{
		R a00{};
		R a01{};
		R a10{};
		R a11{};

		/// <summary>
		/// Returns zero matrix
		/// </summary>
		static Matrix2x2<R> zero()
		{
			return { R(0), R(0), R(0), R(0) };
		}

		/// <summary>
		/// Returns identity matrix
		/// </summary>
		static Matrix2x2<R> identity()
		{
			return { R(1), R(0), R(0), R(1) };
		}

		/// <summary>
		/// Returns random vector
		/// </summary>
		static Matrix2x2<R> random()
		{
			return { R(Utils::get_random(-1, 1)),
					 R(Utils::get_random(-1, 1)),
					 R(Utils::get_random(-1, 1)),
					 R(Utils::get_random(-1, 1)),};
		}

		/// <summary>
		/// Returns transposed matrix
		/// </summary>
		Matrix2x2<R> transpose() const
		{
			return { a00, a10, a01, a11 };
		}

		/// <summary>
		/// Returns determinant of the matrix
		/// </summary>
		R det() const
		{
			return a00 * a11 - a01 * a10;
		}

		/// <summary>
		/// Returns inversed matrix or throws exception is inversed matrix does not exist
		/// </summary>
		Matrix2x2<R> inverse() const
		{
			const auto determinant = det();

			if (std::abs(determinant) < std::numeric_limits<R>::epsilon())
				throw std::exception("The matrix is singular.");

			return Matrix2x2<R>{ a11, -a01, -a10, a00} / determinant;
		}

		/// <summary>
		/// Compound addition operator
		/// </summary>
		Matrix2x2<R>& operator +=(const Matrix2x2<R>& matr)
		{
			a00 += matr.a00;
			a01 += matr.a01;
			a10 += matr.a10;
			a11 += matr.a11;

			return *this;
		}

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		Matrix2x2<R>& operator -=(const Matrix2x2<R>& matr)
		{
			a00 -= matr.a00;
			a01 -= matr.a01;
			a10 -= matr.a10;
			a11 -= matr.a11;

			return *this;
		}

		/// <summary>
		/// Compound multiplication operator
		/// </summary>
		Matrix2x2<R>& operator *=(const Matrix2x2<R>& matr)
		{
			const auto a00_temp = a00 * matr.a00 + a01 * matr.a10;
			const auto a01_temp = a00 * matr.a01 + a01 * matr.a11;
			a00 = a00_temp;
			a01 = a01_temp;

			const auto a10_temp = a10 * matr.a00 + a11 * matr.a10;
			const auto a11_temp = a10 * matr.a01 + a11 * matr.a11;

			a10 = a10_temp;
			a11 = a11_temp;

			return *this;
		}

		/// <summary>
		/// Compound multiplication by scalar operator
		/// </summary>
		Matrix2x2<R>& operator *=(const R& scalar)
		{
			a00 *= scalar;
			a01 *= scalar;
			a10 *= scalar;
			a11 *= scalar;

			return *this;
		}

		/// <summary>
		/// Compound division by scalar operator
		/// </summary>
		Matrix2x2<R>& operator /=(const R& scalar)
		{
			const auto multiplicant = R(1) / scalar;
			return *this *= multiplicant;
		}

		/// <summary>
		/// Returns matrix that represents 2d rotation transformation 
		/// </summary>
		/// <param name="angle">Rotation angle in radians</param>
		/// <returns></returns>
		static Matrix2x2<R> rotation(const R& angle)
		{
			const auto cosine = cos(angle);
			const auto sine = sin(angle);
			return { cosine, -sine, sine, cosine };
		}

		/// <summary>
		/// Returns squared Frobenius norm of the matrix
		/// </summary>
		R norm_sqr() const
		{
			return a00 * a00 + a01 * a01 + a10 * a10 + a11 * a11;
		}

		/// <summary>
		/// Returns Frobenius norm of the matrix
		/// </summary>
		R norm() const
		{
			std::sqrt(norm_sqr());
		}

		/// <summary>
		/// Maximal absolute value of matrix elements
		/// </summary>
		R max_abs() const
		{
			return std::max(std::abs(a00), std::max(std::abs(a01), std::max(std::abs(a10), std::abs(a11))));
		}
	};

	/// <summary>
	/// Matrix-vector multiplication operator
	/// </summary>
	template <class R>
	Vector2d<R> operator *(const Matrix2x2<R>& matr, const Vector2d<R>& vec)
	{
		return { matr.a00 * vec.x + matr.a01 * vec.y,
				 matr.a10 * vec.x + matr.a11 * vec.y };

	}

	/// <summary>
	/// Addition operator
	/// </summary>
	template <class R>
	Matrix2x2<R> operator +(const Matrix2x2<R>& matr1, const Matrix2x2<R>& matr2)
	{
		auto result = matr1;
		return result += matr2;
	}

	/// <summary>
	/// Subtraction operator
	/// </summary>
	template <class R>
	Matrix2x2<R> operator -(const Matrix2x2<R>& matr1, const Matrix2x2<R>& matr2)
	{
		auto result = matr1;
		return result -= matr2;
	}

	/// <summary>
	/// Multiplication operator
	/// </summary>
	template <class R>
	Matrix2x2<R> operator *(const Matrix2x2<R>& matr1, const Matrix2x2<R>& matr2)
	{
		auto result = matr1;
		return result *= matr2;
	}

	/// <summary>
	/// Multiplication by scalar from the right
	/// </summary>
	template <class R>
	Matrix2x2<R> operator *(const Matrix2x2<R>& matr, const R& scalar)
	{
		auto result = matr;
		return result *= scalar;
	}

	/// <summary>
	/// Multiplication by scalar from the left
	/// </summary>
	template <class R>
	Matrix2x2<R> operator *(const R& scalar, const Matrix2x2<R>& matr)
	{
		return matr * scalar;
	}

	/// <summary>
	/// Division by scalar
	/// </summary>
	template <class R>
	Matrix2x2<R> operator /(const Matrix2x2<R>& matr, const R& scalar)
	{
		auto result = matr;
		return result /= scalar;
	}

	/// <summary>
	/// Opposite sign operator
	/// </summary>
	template <class R>
	Matrix2x2<R> operator -(const Matrix2x2<R>& matr)
	{
		return { -matr.a00, -matr.a01, -matr.a10, -matr.a11 };
	}

	/// <summary>
	/// "Equals to" operator
	/// </summary>
	template <class R>
	bool operator ==(const Matrix2x2<R>& matr1, const Matrix2x2<R>& matr2)
	{
		return matr1.a00 == matr2.a00 &&
			matr1.a01 == matr2.a01 &&
			matr1.a10 == matr2.a10 &&
			matr1.a11 == matr2.a11;
	}

	/// <summary>
	/// "Not equal to" operator
	/// </summary>
	template <class R>
	bool operator !=(const Matrix2x2<R>& matr1, const Matrix2x2<R>& matr2)
	{
		return !(matr1 == matr2);
	}

	/// <summary>
	/// A matrix representing an affine transformation in 2d space
	/// Can be thought of a 3x3 matrix of the form
	/// | L  T |
	/// | 0  1 |, where L - is a "linear" part of the affine transformation, represented with a 2x2 matrix
	/// and T - is a "translation" part of the affine transformation represented with a 2d translation vector
	/// </summary>
	template <class R = Real>
	struct MatrixAffine2d
	{
		Matrix2x2<R> linear{};

		Vector2d<R> translation{};

		/// <summary>
		/// Returns a random affine transformation
		/// </summary>
		static MatrixAffine2d<R> random()
		{
			return { Matrix2x2<R>::random(), Vector2d<R>::random() };
		}

		/// <summary>
		/// Returns identity affine transformation
		/// </summary>
		static MatrixAffine2d<R> identity()
		{
			return { Matrix2x2<R>::identity(), Vector2d<R>::zero() };
		}

		/// <summary>
		/// Compound multiplication operator
		/// </summary>
		MatrixAffine2d<R>& operator *=(const MatrixAffine2d<R>& matr)
		{
			translation += linear * matr.translation;
			linear *= matr.linear;

			return *this;
		}

		/// <summary>
		/// Returns inverse affine transformation of throws the exception if the transformation is singular
		/// </summary>
		MatrixAffine2d<R> inverse() const
		{
			const auto linear_inverse = linear.inverse();

			return { linear_inverse, -linear_inverse * translation };
		}

		/// <summary>
		/// Returns rigid motion transformation which represents a rotation around the given center on the given angle
		/// </summary>
		/// <param name="angle">Rotation angle in radians</param>
		/// <param name="center">Center of the rotation</param>
		static MatrixAffine2d<R> build_rotation(const R& angle, const Vector2d<Real>& center)
		{
			const auto rotation = Matrix2x2<Real>::rotation(angle);
			return MatrixAffine2d<R>{rotation, center - rotation * center};
		}

		/// <summary>
		/// Returns rigid motion matrix representing translation on the given vector
		/// </summary>
		static MatrixAffine2d<R> build_translation(const Vector2d<Real>& translation_vect)
		{
			return MatrixAffine2d<R>{Matrix2x2<Real>::identity(), translation_vect};
		}
	};

	/// <summary>
	/// Vector affine transformation
	/// </summary>
	template <class R>
	Vector2d<R> operator *(const MatrixAffine2d<R>& matr, const Vector2d<R>& vec)
	{
		return matr.linear * vec + matr.translation;
	}

	/// <summary>
	/// Multiplication operator
	/// </summary>
	template <class R>
	MatrixAffine2d<R> operator *(const MatrixAffine2d<R>& matr1, const MatrixAffine2d<R>& matr2)
	{
		auto result = matr1;
		return result *= matr2;
	}

	using Index2d = Vector2d<long long>;
}
