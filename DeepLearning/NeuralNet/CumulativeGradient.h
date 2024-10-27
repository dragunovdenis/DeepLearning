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

#include "LayerGradient.h"
#include "../Math/LinAlg3d.h"

namespace DeepLearning
{
	/// <summary>
	/// Represent a gradient of a cost function with respect to weights and biases of a single neuron layer
	/// The cost function is assumed to be an average of some "partial" cost function evaluated for
	/// different input values of the neural network input. So that the gradient itself is an average of
	/// the gradients of the "partial" cost function. The structure below allows to accumulate impact of each 
	/// particular input and then calculate the average on demand
	/// </summary>
	template <class D>
	class CumulativeGradient
	{
		/// <summary>
		///	The gradient
		/// </summary>
		LayerGradient<D> _gradient_sum{};

		/// <summary>
		/// Number of the items accumulated in the corresponding sums
		/// </summary>
		std::size_t _accumulated_items_count{};

	public:

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="weight_tensor_size">Size of a single weight kernel (filter).
		/// Number of filters (channels) can be derived from the number of layers (channels) in the tensor of biases</param>
		/// <param name="bias_tensor_size">Size of the tensor of biases</param>
		CumulativeGradient(const Index3d& weight_tensor_size, const Index3d& bias_tensor_size);

		/// <summary>
		/// Adds given gradients to the corresponding "sum" structures
		/// </summary>
		void add(const LayerGradient<D>& gradient);

		/// <summary>
		/// Calculates and returns the "average" gradient with respect to layer weights and biases
		/// </summary>
		LayerGradient<D> calc_average_gradient(const Real scale_factor = static_cast<Real>(1)) const;

		/// <summary>
		/// Returns reference to the accumulated gradient (sum of all added gradients)
		/// </summary>
		LayerGradient<D>& get_gradient_sum();

		/// <summary>
		/// Returns number of accumulated items
		/// </summary>
		std::size_t items_count() const;

		/// <summary>
		/// Resets the cumulative structure
		/// </summary>
		void reset();
	};
}

#include "CumulativeGradient.inl"