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
#include <functional>
#include <memory>
#include "../defs.h"
#include "Dual.h"

namespace DeepLearning
{
	/// <summary>
	/// A differentiable function of a single variable,
	/// i.e. a function that can be either evaluated itself or evaluated simultaneously with its derivative
	/// The derivative evaluation mechanism is hidden from the caller
	/// Optionally the function can have 1 parameter. Derivative gets evaluated only with respect
	/// to the "argument", not the "parameter"
	/// </summary>
	struct DiffFunc
	{
	private:
		using func_dual_t = std::function < dual<Real>(dual<Real>, Real)>;
		using func_t = std::function <Real(Real, Real)>;

		func_dual_t func_dual{};
		func_t func{};

		/// <summary>
		/// Default constructor. Keep it private to ensure that the only way
		/// to create an instance of the class is through the factory method below
		/// </summary>
		DiffFunc() = default;

		/// <summary>
		/// Two parameter constructor
		/// </summary>
		DiffFunc(const func_dual_t& f_dual, const func_t& f):func_dual(f_dual), func(f){}

	public:

		/// <summary>
		/// A factory : the only way how an instance of the object can be created
		/// </summary>
		template<class F>
		static std::unique_ptr<DiffFunc> create(const F& func)
		{
			return std::unique_ptr<DiffFunc>(new DiffFunc(func, func));
		}

		/// <summary>
		/// Operator that evaluate the function 
		/// </summary>
		Real operator()(const Real& arg, const Real& param = Real(0)) const;

		/// <summary>
		/// Evaluates the function together with its derivative
		/// </summary>
		std::tuple<Real, Real> calc_funcion_and_derivative(const Real arg, const Real& param = Real(0)) const;
	};
}