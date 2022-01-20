#pragma once

#include "../Math/DenseMatrix.h"
#include "../Math/DenseVector.h"

namespace DeepLearning
{
	/// <summary>
	/// Represent a gradient of a cost function with respect to weights and biases of a single neuron layer
	/// The cost function is assumed to be an average of some "partial" cost function evaluated for
	/// different input values of the neural network input. So that the gradient itself is an average of
	/// the gradients of the "partial" cost function. The structure below allows to accumulate impact of each 
	/// particular input and then calculate the average on demand
	/// </summary>
	class CummulativeLayerGradient
	{
		/// <summary>
		/// Sum of the derivatives with respect to layer weight
		/// </summary>
		DenseMatrix _sum_grad_weights{};

		/// <summary>
		/// Sum of the derivatives with respect to layer biases
		/// </summary>
		DenseVector _sum_grad_biases{};

		/// <summary>
		/// Number of the items accumulated in the corresponding sums
		/// </summary>
		std::size_t _accumulated_items_count{};

	public:

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="in_dim">Input dimension of the corresponding neuron layer</param>
		/// <param name="out_dim">Output dimension of the corresponding neuron layer</param>
		CummulativeLayerGradient(const std::size_t in_dim, const std::size_t out_dim);

		/// <summary>
		/// Adds given gradients to the corresponding "sum" structures
		/// </summary>
		/// <param name="weight_grad">"Partial" gradient with respect to weights</param>
		/// <param name="bias_grad">"Partial" gradient with respect to biases</param>
		void Add(const DenseMatrix& weight_grad, const DenseVector& bias_grad);

		/// <summary>
		/// Calculates and returns the "average" gradient with respect to layer weights and biases
		/// </summary>
		std::tuple<DenseMatrix, DenseVector> calc_average_grarient(const Real scale_factor = Real(1)) const;

		/// <summary>
		/// Resets the cumulative structure
		/// </summary>
		void reset();
	};

	/// <summary>
	/// Auxiliary data to perform learning on a level of single neuron layer
	/// </summary>
	struct AuxLearningData
	{
		/// <summary>
		/// Container to store input of a neuron layer
		/// </summary>
		DenseVector Input{};

		/// <summary>
		/// Container to store derivatives of the activation functions
		/// </summary>
		DenseVector Derivatives{};
	};

}
