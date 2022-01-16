#pragma once
#include <array>

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

		/// <summary>
		/// Square root function
		/// </summary>
		friend dual<R, Dim> sqrt(dual<R, Dim> arg)
		{
			arg.x = sqrt(arg.x);
			arg.scale_dual_part(R(1) / (R(2) * arg.x));

			return arg;
		}
	};
}