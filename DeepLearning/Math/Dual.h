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
#include <array>
#include <cmath>
#include <algorithm>

namespace DeepLearning
{

	template<class R, int Dim = 1>
	class dual
	{
	private:
		/// <summary>
		/// "Real" part of the dual number
		/// </summary>
		R x{};

		/// <summary>
		/// multi dimensional dual part of the "dual" number
		/// </summary>
		std::array<R, Dim> d{};

		/// <summary>
		/// Scales the dual part by the given factor
		/// </summary>
		void scale_dual_part(const R& scale_factor)
		{
			for (int i = 0; i < Dim; i++)
				d[i] *= scale_factor;
		}

	public:
		/// <summary>
		/// Getter the real component
		/// </summary>
		/// <returns></returns>
		R Real() const { return x; }

		/// <summary>
		/// Setter for the real component
		/// </summary>
		/// <returns></returns>
		R& Real() { return x; }

		/// <summary>
		/// Getter for the dual component 
		/// </summary>
		const std::array<R, Dim>& Dual() const
		{
			return d;
		}

		/// <summary>
		/// Setter for the dual component 
		/// </summary>
		/// <returns></returns>
		std::array<R, Dim>& Dual()
		{
			return d;
		}

		dual() = default;

		/// <summary>
		/// Constructor from a scalar type
		/// </summary>
		/// <param name="scalar">The scalar value to initialize the dual number</param>
		dual(const R& scalar) : x{ scalar }, d{}
		{}

		/// <summary>
		/// Constructor for "full" initialization
		/// </summary>
		dual(const R& scalar, const std::array<R, Dim>& dual_array) : x{ scalar }, d{ dual_array }
		{}

		/// <summary>
		/// Composite assignment operator +=
		/// </summary>
		dual<R, Dim>& operator +=(const dual<R, Dim>& rhs)
		{
			x += rhs.x;
			for (int i = 0; i < Dim; i++)
				d[i] += rhs.d[i];

			return *this;
		}

		/// <summary>
		/// Composite assignment operator -=
		/// </summary>
		dual<R, Dim>& operator -=(const dual<R, Dim>& rhs)
		{
			x -= rhs.x;
			for (int i = 0; i < Dim; i++)
				d[i] -= rhs.d[i];

			return *this;
		}

		/// <summary>
		/// Composite assignment operator *=
		/// </summary>
		dual<R, Dim>& operator *=(const dual<R, Dim>& rhs)
		{
			for (int i = 0; i < Dim; i++)
				d[i] = rhs.d[i] * x + d[i] * rhs.x;

			x *= rhs.x;

			return *this;
		}

		/// <summary>
		/// Composite assignment operator /=
		/// </summary>
		dual<R, Dim>& operator /=(const dual<R, Dim>& rhs)
		{
			const auto denom = R(1) / (rhs.x);
			const auto  denom_sqr = denom * denom;
			for (int i = 0; i < Dim; i++)
				d[i] = d[i] * denom - x * rhs.d[i] * denom_sqr;

			x *= denom;

			return *this;
		}

		/// <summary>
		/// Unary minus operator
		/// </summary>
		friend dual<R, Dim> operator -(dual<R, Dim> arg)
		{
			arg.x = -arg.x;
			arg.scale_dual_part(R(-1));

			return arg;
		}

		/// <summary>
		/// Binary "+" operator
		/// </summary>
		friend dual<R, Dim> operator +(dual<R, Dim> lhs, const dual<R, Dim>& rhs)
		{
			return lhs += rhs;
		}

		/// <summary>
		/// Binary "-" operator
		/// </summary>
		friend dual<R, Dim> operator -(dual<R, Dim> lhs, const dual<R, Dim>& rhs)
		{
			return lhs -= rhs;
		}

		/// <summary>
		/// Binary "*" operator
		/// </summary>
		friend dual<R, Dim> operator *(dual<R, Dim> lhs, const dual<R, Dim>& rhs)
		{
			return lhs *= rhs;
		}

		/// <summary>
		/// Binary "/" operator
		/// </summary>
		friend dual<R, Dim> operator /(dual<R, Dim> lhs, const dual<R, Dim>& rhs)
		{
			return lhs /= rhs;
		}

		/// <summary>
		/// Sin function
		/// </summary>
		friend dual<R, Dim> sin(dual<R, Dim> arg)
		{
			arg.scale_dual_part(cos(arg.x));
			arg.x = sin(arg.x);

			return arg;
		}

		/// <summary>
		/// Sin function
		/// </summary>
		friend dual<R, Dim> cos(dual<R, Dim> arg)
		{
			arg.scale_dual_part(-sin(arg.x));
			arg.x = cos(arg.x);

			return arg;
		}

		/// <summary>
		/// Natural logarithm function
		/// </summary>
		friend dual<R, Dim> log(dual<R, Dim> arg)
		{
			arg.scale_dual_part(R(1)/(arg.x));
			arg.x = log(arg.x);

			return arg;
		}

		/// <sumary>
		/// Exponent function
		/// </summary>
		friend dual<R, Dim> exp(dual<R, Dim> arg)
		{
			arg.x = exp(arg.x);
			arg.scale_dual_part(arg.x);

			return arg;
		}

		///<summary>
		/// Hyperbolic sine function
		/// </summary>
		friend dual<R, Dim> sinh(dual<R, Dim> arg)
		{
			arg.scale_dual_part(cosh(arg.x));
			arg.x = sinh(arg.x);

			return arg;
		}

		///<summary>
		/// Hyperbolic cosine function
		/// </summary>
		friend dual<R, Dim> cosh(dual<R, Dim> arg)
		{
			arg.scale_dual_part(sinh(arg.x));
			arg.x = cosh(arg.x);

			return arg;
		}

		///<summary>
		/// Hyperbolic tangent function
		/// </summary>
		friend dual<R, Dim> tanh(dual<R, Dim> arg)
		{
			const auto temp = R(1) / cosh(arg.x);
			arg.scale_dual_part(temp * temp);
			arg.x = tanh(arg.x);

			return arg;
		}

		/// <summary>
		/// Square root function
		/// </summary>
		friend dual<R, Dim> sqrt(dual<R, Dim> arg)
		{
			arg.x = sqrt(arg.x);
			arg.scale_dual_part(R(1) / (R(2) * arg.x));

			return arg;
		}

		/// <summary>
		/// "Less than" operator
		/// </summary>
		bool operator <(const dual<R, Dim>& arg) const
		{
			return x < arg.x;
		}

		/// <summary>
		/// "Less or equal" operator
		/// </summary>
		bool operator <=(const dual<R, Dim>& arg) const
		{
			return x <= arg.x;
		}

		/// <summary>
		/// "Greater than" operator
		/// </summary>
		bool operator >(const dual<R, Dim>& arg) const
		{
			return !(*this <= arg.x);
		}

		/// <summary>
		/// "Greater or equal" operator
		/// </summary>
		bool operator >=(const dual<R, Dim>& arg) const
		{
			return !(*this < arg.x);
		}
	};
}

namespace std
{
	/// <summary>
	/// Returns true if the given dual number has infinite components
	/// </summary>
	template<class R, int Dim>
	bool isinf(const DeepLearning::dual<R, Dim>& val)
	{
		return std::isinf(val.Real()) || std::any_of(val.Dual().begin(), val.Dual().end(), [](const auto& x) { return std::isinf(x); });
	}

	 //<summary>
	 //Returns true if the given dual number has "not a number" components
	 //</summary>
	template<class R, int Dim>
	bool isnan(const DeepLearning::dual<R, Dim>& val)
	{
		return std::isnan(val.Real()) || std::any_of(val.Dual().begin(), val.Dual().end(), [](const auto& x) { return std::isnan(x); });
	}

	//Definitions of the "numeric limits" properties for the "dual' class
	template<class R, int Dim>
	class numeric_limits<DeepLearning::dual<R, Dim>>
	{
	public:
		/// <summary>
		/// Return "max" value for the current "dual" type
		/// </summary>
		static DeepLearning::dual<R, Dim> max()
		{
			return DeepLearning::dual<R, Dim>(std::numeric_limits<R>::max());
		}
	};
}