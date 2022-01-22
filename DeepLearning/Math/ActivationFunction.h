#pragma once
#include "DenseVector.h"
#include "DiffFunc.h"

namespace DeepLearning
{
	/// <summary>
	/// Identifiers of different activation functions
	/// </summary>
	enum ActivationFunctionId: unsigned int {
		UNKNOWN = 0,
		SIGMOID = 1,
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
		virtual DenseVector operator ()(const DenseVector& input) const;

		/// <summary>
		/// Calculates function and derivative with respect to the given input vector
		/// </summary>
		virtual std::tuple<DenseVector, DenseVector> func_and_deriv(const DenseVector& input) const;
	};
}
