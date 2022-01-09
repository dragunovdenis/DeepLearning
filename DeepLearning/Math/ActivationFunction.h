#pragma once
#include "DenseVector.h"

namespace DeepLearning
{
	/// <summary>
	/// Interface for an activation function
	/// </summary>
	class ActivationFuncion
	{
	public:
		/// <summary>
		/// The function
		/// </summary>
		virtual DenseVector operator ()(const DenseVector& input) const = 0;

		/// <summary>
		/// Calculates function and derivative of the given input vector
		/// </summary>
		virtual std::tuple<DenseVector, DenseVector> func_and_deriv(const DenseVector& input) const = 0;
	};

	/// <summary>
	/// The "sigmoid" activation function
	/// </summary>
	class Sigmoid : public ActivationFuncion
	{
	public:
		/// <summary>
		/// The function
		/// </summary>
		virtual DenseVector operator ()(const DenseVector& input) const;

		/// <summary>
		/// Calculates function and derivative of the given input vector
		/// </summary>
		virtual std::tuple<DenseVector, DenseVector> func_and_deriv(const DenseVector& input) const;
	};
}
