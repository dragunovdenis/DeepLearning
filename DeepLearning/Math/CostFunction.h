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

#include "../CudaBridge.h"
#include "BasicCollection.h"
#include "Dual.h"
#include "Functions.h"

namespace DeepLearning
{
	class BasicCudaCollection;

	/// <summary>
	/// Identifiers of different cost functions
	/// </summary>
	enum class CostFunctionId : unsigned int
	{
		UNKNOWN = 0,
		SQUARED_ERROR = 1,
		CROSS_ENTROPY = 2,
	};

	/// <summary>
	/// Returns string representation of the given cost function type identifier
	/// </summary>
	std::string to_string(const CostFunctionId& cost_type_id);

	/// <summary>
	/// Parses given string to the cost function identifier
	/// </summary>
	CostFunctionId parse_cost_type(const std::string& str);

	/// <summary>
	/// Helper methods used for activation function evaluation
	/// </summary>
	namespace CostFunctionHelper
	{
		/// <summary>
		/// Factory method to instantiate cost functions by their identifiers
		/// </summary>
		template <class F>
		CUDA_CALLABLE F make(const CostFunctionId id)
		{
			switch (id)
			{
			case CostFunctionId::SQUARED_ERROR: return [](const auto& x, const auto& ref)
				{
					const auto diff = x - ref;
					return Real(0.5) * diff * diff;
				};
			case CostFunctionId::CROSS_ENTROPY:return [](const auto& x, const auto& ref)
				{
					if (ref <= Real(0))
						return Func::nan_to_num(-log(Real(1) - x));

					if (ref >= Real(1))
						return Func::nan_to_num(-log(x));

					return Func::nan_to_num(-(ref * log(x) + (Real(1) - ref) * log(Real(1) - x)));
				};
			default: return [](const auto& x, const auto& ref) { return decltype(x)(std::numeric_limits<Real>::signaling_NaN()); };
			}
		}

		/// <summary>
		/// Evaluates the given cost function (represented with its identifier) for the given "output" and "reference" collections
		/// </summary>
		Real evaluate_cost(const BasicCollection& output, const BasicCollection& reference, const CostFunctionId id);

		/// <summary>
		/// Evaluates the given cost function (represented with its identifier) for the given "output" and "reference" collections and
		/// calculates gradient of the cost function with respect to the elements of the "output" collection
		/// </summary>
		/// <param name="output">Output of a neural network in training; used to store the calculated gradient of the cost function</param>
		/// <param name="reference">Reference vector for the given output</param>
		/// <param name="id">Identifier of the cost function</param>
		Real evaluate_cost_and_gradient(BasicCollection& output, const BasicCollection& reference, const CostFunctionId id);

		/// <summary>
		/// Evaluates gradient of the given cost function (represented with its identifier) for the given "output" and "reference" collections 
		/// with respect to the elements of the "output" collection
		/// </summary>
		/// <param name="output">Output of a neural network in training; used to store the calculated gradient of the cost function</param>
		/// <param name="reference">Reference vector for the given output</param>
		/// <param name="id">Identifier of the cost function</param>
		void evaluate_gradient(BasicCollection& output, const BasicCollection& reference, const CostFunctionId id);

		/// <summary>
		/// Evaluates the given cost function (represented with its identifier) for the given "output" and "reference" collections
		/// </summary>
		Real evaluate_cost(const BasicCudaCollection& output, const BasicCudaCollection& reference, const CostFunctionId id);

		/// <summary>
		/// Evaluates the given cost function (represented with its identifier) for the given "output" and "reference" collections and
		/// calculates gradient of the cost function with respect to the elements of the "output" collection
		/// </summary>
		/// <param name="output">Output of a neural network in training; used to store the calculated gradient of the cost function</param>
		/// <param name="reference">Reference vector for the given output</param>
		/// <param name="id">Identifier of the cost function</param>
		Real evaluate_cost_and_gradient(BasicCudaCollection& output, const BasicCudaCollection& reference, const CostFunctionId id);

		/// <summary>
		/// Evaluates gradient of the given cost function (represented with its identifier) for the given "output" and "reference" collections 
		/// with respect to the elements of the "output" collection
		/// </summary>
		/// <param name="output">Output of a neural network in training; used to store the calculated gradient of the cost function</param>
		/// <param name="reference">Reference vector for the given output</param>
		/// <param name="id">Identifier of the cost function</param>
		void evaluate_gradient(BasicCudaCollection& output, const BasicCudaCollection& reference, const CostFunctionId id);
	}

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
	};
}
