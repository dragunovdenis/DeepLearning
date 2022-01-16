#pragma once

#include "DiffFunc.h"
#include "DenseVector.h"

namespace DeepLearning
{
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
	/// Representation of the cost function that is used in the neural networks learning process
	/// </summary>
	class CostFunction
	{
		std::unique_ptr<DiffFunc> _func;
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
		virtual DenseVector operator ()(const DenseVector& output, const DenseVector& reference) const;

		/// <summary>
		/// Calculates function and derivative with respect to the given output vector
		/// </summary>
		/// <param name="output">Actual output of the neural network</param>
		/// <param name="reference">Expected output of the neural network</param>
		virtual std::tuple<DenseVector, DenseVector> func_and_deriv(const DenseVector& output, const DenseVector& reference) const;




	};
}
