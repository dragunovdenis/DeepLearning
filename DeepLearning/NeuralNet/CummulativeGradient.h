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

#include "../Math/Matrix.h"
#include "../Math/Vector.h"
#include "../Math/Tensor.h"
#include <vector>

namespace DeepLearning
{
	/// <summary>
	/// Represent a gradient of a cost function with respect to weights and biases of a single neuron layer
	/// The cost function is assumed to be an average of some "partial" cost function evaluated for
	/// different input values of the neural network input. So that the gradient itself is an average of
	/// the gradients of the "partial" cost function. The structure below allows to accumulate impact of each 
	/// particular input and then calculate the average on demand
	/// </summary>
	class CummulativeGradient
	{
		/// <summary>
		/// Sum of the derivatives with respect to layer weights
		/// </summary>
		std::vector<Tensor> _sum_grad_weights{};

		/// <summary>
		/// Sum of the derivatives with respect to layer biases
		/// </summary>
		Tensor _sum_grad_biases{};

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
		CummulativeGradient(const std::size_t in_dim, const std::size_t out_dim);

		/// <summary>
		/// Adds given gradients to the corresponding "sum" structures
		/// </summary>
		/// <param name="weight_grad">"Partial" gradient with respect to weights</param>
		/// <param name="bias_grad">"Partial" gradient with respect to biases</param>
		void Add(const std::vector<Tensor>& weight_grad, const Tensor& bias_grad);

		/// <summary>
		/// Calculates and returns the "average" gradient with respect to layer weights and biases
		/// </summary>
		std::tuple<std::vector<Tensor>, Tensor> calc_average_grarient(const Real scale_factor = Real(1)) const;

		/// <summary>
		/// Resets the cumulative structure
		/// </summary>
		void reset();
	};
}
