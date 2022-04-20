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
#include "DiffFunc.h"

namespace DeepLearning
{
	/// <summary>
	/// Identifiers of different activation functions
	/// </summary>
	enum class ActivationFunctionId: unsigned int {
		UNKNOWN = 0,
		SIGMOID = 1, //Sigmoid activation function
		TANH = 2, //Hyperbolic tangent activation function
		RELU = 3, // rectified linear activation (unit)
	};

	/// <summary>
	/// Interface for an activation function
	/// </summary>
	class ActivationFuncion
	{
		std::unique_ptr<DiffFunc> _func{};

	public:

		/// <summary>
		/// Constructor
		/// </summary>
		ActivationFuncion(const ActivationFunctionId id);

		/// <summary>
		/// The function
		/// </summary>
		template <class T>
		T operator ()(const T& input) const;

		/// <summary>
		/// Calculates function and derivative with respect to the given input vector
		/// </summary>
		template <class T>
		std::tuple<T, T> func_and_deriv(const T& input) const;
	};
}
