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
#include "../CudaBridge.h"

namespace DeepLearning
{
	/// <summary>
	/// A functionality that allows calculating derivatives via the "automatic differentiation"
	/// approach in its " forward accumulation" version (see https://en.wikipedia.org/wiki/Automatic_differentiation for more detail)
	/// </summary>
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
		R  d[Dim]{};

		/// <summary>
		/// Scales the dual part by the given factor
		/// </summary>
		CUDA_CALLABLE void scale_dual_part(const R& scale_factor)
		{
			for (int i = 0; i < Dim; i++)
				d[i] *= scale_factor;
		}

	public:
		/// <summary>
		/// Getter the real component
		/// </summary>
		/// <returns></returns>
		CUDA_CALLABLE R Real() const { return x; }

		/// <summary>
		/// Setter for the real component
		/// </summary>
		/// <returns></returns>
		CUDA_CALLABLE R& Real() { return x; }

		/// <summary>
		/// Getter for the dual component 
		/// </summary>
		CUDA_CALLABLE const R* Dual() const
		{
			return d;
		}

		/// <summary>
		/// Setter for the dual component 
		/// </summary>
		/// <returns></returns>
		CUDA_CALLABLE R* Dual()
		{
			return d;
		}

		dual() = default;

		/// <summary>
		/// Constructor from a scalar type
		/// </summary>
		/// <param name="scalar">The scalar value to initialize the dual number</param>
		CUDA_CALLABLE dual(const R& scalar) : x{ scalar }, d{}
		{}

		/// <summary>
		/// Constructor for "full" initialization
		/// </summary>
		dual(const R& scalar, const std::array<R, Dim>& dual_array) : x{ scalar }
		{
			std::copy(dual_array.begin(), dual_array.end(), d);
		}

		/// <summary>
		/// Special constructor for the case when "dual dimension" is equal to "1"
		/// </summary>
		CUDA_CALLABLE dual(const R& scalar, const R& dual) : x{ scalar }
		{
			static_assert(Dim == 1, "This constructor is supposed to be called only when dual dimension is equal to 1");
			d[0] = dual;
		}

		/// <summary>
		/// Composite assignment operator +=
		/// </summary>
		CUDA_CALLABLE dual<R, Dim>& operator +=(const dual<R, Dim>& rhs)
		{
			x += rhs.x;
			for (int i = 0; i < Dim; i++)
				d[i] += rhs.d[i];

			return *this;
		}

		/// <summary>
		/// Composite assignment operator -=
		/// </summary>
		CUDA_CALLABLE dual<R, Dim>& operator -=(const dual<R, Dim>& rhs)
		{
			x -= rhs.x;
			for (int i = 0; i < Dim; i++)
				d[i] -= rhs.d[i];

			return *this;
		}

		/// <summary>
		/// Composite assignment operator *=
		/// </summary>
		CUDA_CALLABLE dual<R, Dim>& operator *=(const dual<R, Dim>& rhs)
		{
			for (int i = 0; i < Dim; i++)
				d[i] = rhs.d[i] * x + d[i] * rhs.x;

			x *= rhs.x;

			return *this;
		}

		/// <summary>
		/// Composite assignment operator /=
		/// </summary>
		CUDA_CALLABLE dual<R, Dim>& operator /=(const dual<R, Dim>& rhs)
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
		CUDA_CALLABLE friend dual<R, Dim> operator -(dual<R, Dim> arg)
		{
			arg.x = -arg.x;
			arg.scale_dual_part(R(-1));

			return arg;
		}

		/// <summary>
		/// Binary "+" operator
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> operator +(dual<R, Dim> lhs, const dual<R, Dim>& rhs)
		{
			return lhs += rhs;
		}

		/// <summary>
		/// Binary "-" operator
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> operator -(dual<R, Dim> lhs, const dual<R, Dim>& rhs)
		{
			return lhs -= rhs;
		}

		/// <summary>
		/// Binary "*" operator
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> operator *(dual<R, Dim> lhs, const dual<R, Dim>& rhs)
		{
			return lhs *= rhs;
		}

		/// <summary>
		/// Binary "/" operator
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> operator /(dual<R, Dim> lhs, const dual<R, Dim>& rhs)
		{
			return lhs /= rhs;
		}

		/// <summary>
		/// Sin function
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> sin(dual<R, Dim> arg)
		{
			arg.scale_dual_part(std::cos(arg.x));
			arg.x = std::sin(arg.x);

			return arg;
		}

		/// <summary>
		/// Sin function
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> cos(dual<R, Dim> arg)
		{
			arg.scale_dual_part(-std::sin(arg.x));
			arg.x = std::cos(arg.x);

			return arg;
		}

		/// <summary>
		/// Natural logarithm function
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> log(dual<R, Dim> arg)
		{
			arg.scale_dual_part(R(1)/(arg.x));
			arg.x = std::log(arg.x);

			return arg;
		}

		/// <sumary>
		/// Exponent function
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> exp(dual<R, Dim> arg)
		{
			arg.x = std::exp(arg.x);
			arg.scale_dual_part(arg.x);

			return arg;
		}

		///<summary>
		/// Hyperbolic sine function
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> sinh(dual<R, Dim> arg)
		{
			arg.scale_dual_part(std::cosh(arg.x));
			arg.x = std::sinh(arg.x);

			return arg;
		}

		///<summary>
		/// Hyperbolic cosine function
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> cosh(dual<R, Dim> arg)
		{
			arg.scale_dual_part(std::sinh(arg.x));
			arg.x = std::cosh(arg.x);

			return arg;
		}

		///<summary>
		/// Hyperbolic tangent function
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> tanh(dual<R, Dim> arg)
		{
			const auto temp = R(1) / std::cosh(arg.x);
			arg.scale_dual_part(temp * temp);
			arg.x = std::tanh(arg.x);

			return arg;
		}

		/// <summary>
		/// Square root function
		/// </summary>
		CUDA_CALLABLE friend dual<R, Dim> sqrt(dual<R, Dim> arg)
		{
			arg.x = std::sqrt(arg.x);
			arg.scale_dual_part(R(1) / (R(2) * arg.x));

			return arg;
		}

		/// <summary>
		/// "Less than" operator
		/// </summary>
		CUDA_CALLABLE bool operator <(const dual<R, Dim>& arg) const
		{
			return x < arg.x;
		}

		/// <summary>
		/// "Less or equal" operator
		/// </summary>
		CUDA_CALLABLE bool operator <=(const dual<R, Dim>& arg) const
		{
			return x <= arg.x;
		}

		/// <summary>
		/// "Greater than" operator
		/// </summary>
		CUDA_CALLABLE bool operator >(const dual<R, Dim>& arg) const
		{
			return !(*this <= arg.x);
		}

		/// <summary>
		/// "Greater or equal" operator
		/// </summary>
		CUDA_CALLABLE bool operator >=(const dual<R, Dim>& arg) const
		{
			return !(*this < arg.x);
		}
	};
}

/// <summary>
/// Returns true if the given dual number has infinite components
/// </summary>
template<class R>
CUDA_CALLABLE bool isinf(const DeepLearning::dual<R, 1>& val)
{
	return isinf(val.Real()) || isinf(val.Dual()[0]);
}

/// <summary>
/// Returns true if the given dual number has "not a number" components
/// </summary>
template<class R>
CUDA_CALLABLE bool isnan(const DeepLearning::dual<R, 1>& val)
{
	return isnan(val.Real()) || isnan(val.Dual()[0]);
}

namespace std
{
	/// <summary>
	/// Returns true if the given dual number has infinite components
	/// </summary>
	template<class R, int Dim>
	inline bool isinf(const DeepLearning::dual<R, Dim>& val)
	{
		return std::isinf(val.Real()) || std::any_of(val.Dual(), val.Dual() + Dim, [](const auto& x) { return std::isinf(x); });
	}

	 //<summary>
	 //Returns true if the given dual number has "not a number" components
	 //</summary>
	template<class R, int Dim>
	inline bool isnan(const DeepLearning::dual<R, Dim>& val)
	{
		return std::isnan(val.Real()) || std::any_of(val.Dual(), val.Dual() + Dim, [](const auto& x) { return std::isnan(x); });
	}

	//Definitions of the "numeric limits" properties for the "dual' class
	template<class R, int Dim>
	class numeric_limits<DeepLearning::dual<R, Dim>>
	{
	public:
		/// <summary>
		/// Return "max" value for the current "dual" type
		/// </summary>
		CUDA_CALLABLE static DeepLearning::dual<R, Dim> max()
		{
			return DeepLearning::dual<R, Dim>(std::numeric_limits<R>::max());
		}
	};
}