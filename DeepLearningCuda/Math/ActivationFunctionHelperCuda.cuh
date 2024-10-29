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
#include <Math/ActivationFunctionId.h>

namespace DeepLearning
{
	class BasicCudaCollection;
}

namespace DeepLearning
{
	/// <summary>
	/// Facilitates evaluation of different activation functions.
	/// </summary>
	class ActivationFunctionHelperCuda
	{
	public:

		/// <summary>
		/// Evaluates given function at each element of the given collection and stores the result "in place"
		/// </summary>
		static void evaluate_in_place(BasicCudaCollection& collection, const ActivationFunctionId id);

		/// <summary>
		/// Evaluates given function and its derivative at each element of the given "collection_func"
		/// Stores the function value to the "collection_func" whereas the derivative value is stored to the "collection_deriv"
		/// </summary>
		static void evaluate_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv, const ActivationFunctionId id);

		/// <summary>
		/// Subtracts maximal element in the given collection from each element of the collection,
		/// evaluated exponent of each element and sore the result to the given collection
		/// </summary>
		static void normalize_and_evaluate_exponent_in_place(BasicCudaCollection& collection);

		/// <summary>
		/// Evaluates gradient of the soft-max function with respect to its input
		/// </summary>
		/// <param name="input_exp">Collection containing exponents of the soft-max input</param>
		/// <param name="result">Placeholder for the calculated gradient. Should be allocated by the caller
		/// Must be initialized with a copy of "input_exp" by the caller</param>
		static void evaluate_softmax_input_grad(const BasicCudaCollection& input_exp, const BasicCudaCollection& out_grad, BasicCudaCollection& result);

		/// <summary>
		/// Evaluates ReLu function at each value of the given collection "in-place".
		/// </summary>
		static void relu_in_place(BasicCudaCollection& collection);

		/// <summary>
		/// Evaluates ReLu function and its derivative at each value of the given collection
		/// "in-place" (the derivative is stored in the other given collection-container).
		/// </summary>
		static void relu_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv);

		/// <summary>
		/// Evaluates Sigmoid function at each value of the given collection "in-place".
		/// </summary>
		static void sigmoid_in_place(BasicCudaCollection& collection);

		/// <summary>
		/// Evaluates Sigmoid function and its derivative at each value of the given collection
		/// "in-place" (the derivative is stored in the other given collection-container).
		/// </summary>
		static void sigmoid_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv);

		/// <summary>
		/// Evaluates Tanh function at each value of the given collection "in-place".
		/// </summary>
		static void tanh_in_place(BasicCudaCollection& collection);

		/// <summary>
		/// Evaluates Tanh function and its derivative at each value of the given collection
		/// "in-place" (the derivative is stored in the other given collection-container).
		/// </summary>
		static void tanh_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv);
	};
}
