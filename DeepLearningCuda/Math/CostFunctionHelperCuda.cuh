//Copyright (c) 2024 Denys Dragunov, dragunovdenis@gmail.com
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
#include <Math/CostFunctionId.h>
#include <defs.h>

/// <summary>
	/// Helper methods used for activation function evaluation
	/// </summary>
namespace DeepLearning
{
	class BasicCudaCollection;

	/// <summary>
	/// Functionality that facilitates evaluation of cost function.
	/// </summary>
	class CostFunctionHelperCuda
	{
	public:

		/// <summary>
		/// Evaluates the given cost function (represented with its identifier) for the given "output" and "reference" collections
		/// </summary>
		static Real evaluate_cost(const BasicCudaCollection& output, const BasicCudaCollection& reference, const CostFunctionId id);

		/// <summary>
		/// Evaluates the given cost function (represented with its identifier) for the given "output" and "reference" collections and
		/// calculates gradient of the cost function with respect to the elements of the "output" collection
		/// </summary>
		/// <param name="output">Output of a neural network in training; used to store the calculated gradient of the cost function</param>
		/// <param name="reference">Reference vector for the given output</param>
		/// <param name="id">Identifier of the cost function</param>
		static Real evaluate_cost_and_gradient(BasicCudaCollection& output, const BasicCudaCollection& reference, const CostFunctionId id);

		/// <summary>
		/// Evaluates gradient of the given cost function (represented with its identifier) for the given "output" and "reference" collections 
		/// with respect to the elements of the "output" collection
		/// </summary>
		/// <param name="output">Output of a neural network in training; used to store the calculated gradient of the cost function</param>
		/// <param name="reference">Reference vector for the given output</param>
		/// <param name="id">Identifier of the cost function</param>
		static void evaluate_gradient(BasicCudaCollection& output, const BasicCudaCollection& reference, const CostFunctionId id);
	};
}
