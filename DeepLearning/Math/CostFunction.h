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

#include "CostFunctionId.h"

namespace DeepLearning
{
	/// <summary>
	/// Representation of the cost function that is used in the neural networks learning process
	/// </summary>
	template <class T>
	class CostFunction
	{
		const CostFunctionId _id;
	public:

		/// <summary>
		/// Constructor
		/// </summary>
		CostFunction(const CostFunctionId id);

		/// <summary>
		/// The function
		/// </summary>
		/// <param name="output">Actual output of the neural network</param>
		/// <param name="reference">Expected output of the neural network</param>
		Real operator ()(const T& output, const T& reference) const;

		/// <summary>
		/// Calculates function and derivative with respect to the given "output" vector
		/// </summary>
		/// <param name="output">Actual output of the neural network</param>
		/// <param name="reference">Expected output of the neural network</param>
		std::tuple<Real, T> func_and_deriv(const T& output, const T& reference) const;

		/// <summary>
		/// Calculates derivative of the cost function with respect to the given "output" vector
		/// </summary>
		/// <param name="output">Actual output of the neural network</param>
		/// <param name="reference">Expected output of the neural network</param>
		T deriv(const T& output, const T& reference) const;

		/// <summary>
		/// Calculates derivative of the cost function with respect to the given "output" vector
		/// </summary>
		/// <param name="output_deriv">Actual output of the neural network.
		/// Contains the derivatives when method is returned</param>
		/// <param name="reference">Expected output of the neural network</param>
		void deriv_in_place(T& output_deriv, const T& reference) const;
	};
}

#include "CostFunction.inl"