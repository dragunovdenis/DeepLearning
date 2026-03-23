//Copyright (c) 2026 Denys Dragunov, dragunovdenis@gmail.com
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

#include "CppUnitTest.h"
#include <NeuralNet/NLayer.h>
#include <NeuralNet/CLayer.h>
#include <NeuralNet/PLayer.h>
#include <Utilities.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	/// <summary>
	/// Generic method to construct a "standard" CLayer for tests.
	/// </summary>
	template <class D>
	CLayer<D> CreateCLayer(const ActivationFunctionId activation)
	{
		const auto input_dim = Index3d(5, 13, 17);
		const auto filter_window = Index2d(3);
		constexpr auto filters_count = 7;
		const auto paddings = Index3d(0, 3, 6);
		const auto strides = Index3d(2);
		return CLayer<D>(input_dim, filter_window, filters_count, activation, paddings, strides);
	}

	/// <summary>
	/// Returns a "standard" PLayer instance to be used in testing.
	/// </summary>
	template <class D>
	PLayer<D> CreatePLayer(const PoolTypeId pool_oper_id)
	{
		const auto input_dim = Index3d(5, 10, 7);
		const auto filter_window = Index2d(3, 4);
		return PLayer<D>(input_dim, filter_window, pool_oper_id);
	}

	/// <summary>
	/// Returns a "standard" NLayer instance to be used in testing.
	/// </summary>
	template <class D>
	NLayer<D> CreateNLayer(const ActivationFunctionId activation_func_id)
	{
		const auto input_dim = 10;
		const auto output_dim = 23;
		return NLayer<D>(input_dim, output_dim, activation_func_id);
	}

	/// <summary>
	/// General method to test gradient with scaling factor calculation.
	/// </summary>
	template <template <typename> class L, class D>
	void RunGeneralGradientWithScalingTest(const L<D>& nl)
	{
		const typename D::tensor_t input(nl.in_size(), static_cast<Real>(-1), static_cast<Real>(1));
		typename D::tensor_t input_grad_result(nl.in_size(), /*fill zeros*/ true);
		auto layer_data = LayerData<D>(input);
		LayerGradient<D> gradient_container;
		nl.allocate(gradient_container, /*fill zeros*/ false);

		for (auto& gradient_item : gradient_container.data)
			gradient_item.standard_random_fill();

		const auto gradient_container_input = gradient_container;
		const auto gradient_scale_factor = Utils::get_random(-1, 1);

		//Act
		const auto output = nl.act(layer_data.Input, &layer_data.Trace);
		nl.backpropagate(output, layer_data, input_grad_result, gradient_container,
			/*evaluate_input_gradient*/ true, gradient_scale_factor);

		// Assert
		const auto [reference_input_grad_result, reference_layer_grad_result] = nl.backpropagate(output, layer_data);
		const auto diff = (gradient_container_input * gradient_scale_factor +
			reference_layer_grad_result - gradient_container).max_abs();
		Logger::WriteMessage((std::string("Gradient discrepancy = ") + Utils::to_string(diff) + '\n').c_str());
		Assert::IsTrue(diff < 10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference");
		const auto input_grad_diff = (reference_input_grad_result - input_grad_result).max_abs();
		Logger::WriteMessage((std::string("Input gradient discrepancy = ") + Utils::to_string(input_grad_diff) + '\n').c_str());
		Assert::IsTrue(input_grad_diff < 10 * std::numeric_limits<Real>::epsilon(),
			L"Input gradient must not be affected by scaling factor");
	}
}
