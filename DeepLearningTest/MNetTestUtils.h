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
#include "Math/CostFunction.h"
#include "NeuralNet/MLayerData.h"
#include "StandardTestUtils.h"

using namespace DeepLearning;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

/// <summary>
/// Utility methods to test multi-net related functionality.
/// </summary>
namespace DeepLearningTest::MNetTestUtils
{
	/// <summary>
	/// Returns an instance of "input data" initialized according to the given set of parameters.
	/// </summary>
	template <class D>
	static MLayerData<D> construct_random_data(const int size, const Index3d& item_size)
	{
		MLayerData<D> result(size);

		for (auto itemId = 0; itemId < size; ++itemId)
		{
			result[itemId].resize(item_size);
			result[itemId].init(FillRandomUniform);
		}

		return result;
	}

	/// <summary>
	/// Returns a collection of the corresponding size filled with the given <paramref name="value"/>.
	/// </summary>
	template <class D>
	static LazyVector<typename D::tensor_t> construct_and_fill_vector(const long long size,
		const Index3d& item_size, const Real value)
	{
		LazyVector<typename D::tensor_t> result(size);

		for (auto itemId = 0; itemId < size; ++itemId)
		{
			result[itemId].resize(item_size);
			result[itemId].fill(value);
		}

		return result;
	}

	/// <summary>
	/// Returns a collection of the corresponding size filled with random values from [0, 1].
	/// </summary>
	template <class D>
	static LazyVector<typename D::tensor_t> construct_random_vector(const long long size, const Index3d& item_size)
	{
		LazyVector<typename D::tensor_t> result(size);

		for (auto itemId = 0; itemId < size; ++itemId)
		{
			result[itemId].resize(item_size);
			result[itemId].init(FillRandomUniform);
		}

		return result;
	}

	/// <summary>
	/// Returns value of cost function <paramref name="cost_func"/> evaluated
	/// on the collections <paramref name="out"/> and <paramref name="ref"/>
	/// </summary>
	template <class D>
	Real evaluate_cost(const CostFunction<typename D::tensor_t>& cost_func,
		const LazyVector<typename D::tensor_t>& out, const LazyVector<typename D::tensor_t>& ref)
	{
		if (out.size() != ref.size())
			throw std::exception("Inconsistent input.");

		Real result{};

		for (auto item_id = 0ull; item_id < out.size(); ++item_id)
			result += cost_func(out[item_id], ref[item_id]);

		return result;
	}

	/// <summary>
	/// Runs a standard verification procedure for checking correctness of
	/// the input gradient calculation for the given <paramref name="layer"/>.
	/// </summary>
	template <class D>
	void run_standard_m_layer_input_gradient_test(const AMLayer<D>& layer, const Real delta, const Real tolerance)
	{
		const auto rec_depth = layer.in_size().w;
		const auto in_item_size = layer.in_size().xyz;
		const auto in_size_plain = in_item_size.coord_prod();
		const auto out_item_size = layer.out_size().xyz;

		const auto out_grad = MNetTestUtils::construct_and_fill_vector<CpuDC>(rec_depth, out_item_size, 1);
		const auto input = MNetTestUtils::construct_random_data<CpuDC>(static_cast<int>(rec_depth), in_item_size);
		auto trace_data = input;
		LazyVector<CpuDC::tensor_t> output;
		LazyVector<CpuDC::tensor_t> in_gradient(rec_depth);
		auto layer_grad = layer.allocate_gradient_container(true /*fill zero*/);

		// Act
		layer.act(input, output, &trace_data);
		layer.backpropagate(out_grad, output, trace_data.Data, in_gradient,
			layer_grad, true /*evaluate input gradient*/);

		// Assert
		Real max_gradient_diff{};
		const auto one_over_double_delta = static_cast<Real>(0.5) / delta;

		for (auto depth_id = 0ull; depth_id < in_gradient.size(); ++depth_id)
		{
			const auto& in_grad_for_depth = in_gradient[depth_id];
			Assert::AreEqual<std::size_t>(in_grad_for_depth.size(), in_size_plain,
				L"Invalid dimension of input gradient vector.");

			for (auto item_id = 0; item_id < in_size_plain; ++item_id)
			{
				auto input_plus_delta = input;
				input_plus_delta[depth_id][item_id] += delta;

				auto input_minus_delta = input;
				input_minus_delta[depth_id][item_id] -= delta;

				LazyVector<CpuDC::tensor_t> output_plus_delta;
				layer.act(input_plus_delta, output_plus_delta, nullptr);
				LazyVector<CpuDC::tensor_t> output_minus_delta;
				layer.act(input_minus_delta, output_minus_delta, nullptr);
				const auto in_grad_reference = (output_plus_delta.sum() - output_minus_delta.sum()) * one_over_double_delta;
				const auto in_grad_actual = in_grad_for_depth[item_id];
				const auto diff = std::abs(in_grad_reference - in_grad_actual);
				max_gradient_diff = std::max(max_gradient_diff, diff);
			}
		}

		StandardTestUtils::LogAndAssertLessOrEqualTo(
			"Deviation between actual and expected gradients with respect to the input vector",
			max_gradient_diff, tolerance);
	}

	/// <summary>
	/// Runs a standard verification procedure for checking correctness
	/// of the parameter gradients of the given <paramref name="layer"/>.
	/// </summary>
	template <class L>
	void run_standard_m_layer_parameter_gradient_test(const L& layer, const int param_container_id,
		const Real delta, const Real tolerance)
	{
		const auto rec_depth = layer.in_size().w;
		const auto in_item_size = layer.in_size().xyz;
		const auto out_item_size = layer.out_size().xyz;
		const auto out_grad = MNetTestUtils::construct_and_fill_vector<CpuDC>(rec_depth, out_item_size, 1);
		const auto input = MNetTestUtils::construct_random_data<CpuDC>(static_cast<int>(rec_depth), in_item_size);
		auto trace_data = input;
		LazyVector<CpuDC::tensor_t> in_gradient{};
		auto layer_grad = layer.allocate_gradient_container(true /*fill zero*/);

		// Act
		LazyVector<CpuDC::tensor_t> output;
		layer.act(input, output, &trace_data);
		layer.backpropagate(out_grad, output, trace_data.Data, in_gradient,
			layer_grad, false /*evaluate input gradient*/);

		// Assert
		Real max_gradient_diff{};
		const auto one_over_double_delta = static_cast<Real>(0.5) / delta;
		const auto zero_gradient = layer.allocate_gradient_container(true /*fill zero*/);
		const auto& param_container_grad = layer_grad[0].data[param_container_id];

		for (auto item_id = 0; item_id < param_container_grad.size(); ++item_id)
		{
			auto zero_gradient_plus_delta = zero_gradient;
			zero_gradient_plus_delta[0].data[param_container_id][item_id] = delta;

			auto layer_plus_delta = layer;
			layer_plus_delta.update(zero_gradient_plus_delta, static_cast<Real>(1), static_cast<Real>(0) /*reg factor*/);

			LazyVector<CpuDC::tensor_t> output_plus_delta;
			layer_plus_delta.act(input, output_plus_delta, nullptr);

			auto layer_minus_delta = layer;
			layer_minus_delta.update(zero_gradient_plus_delta, static_cast<Real>(-1), static_cast<Real>(0) /*reg factor*/);

			LazyVector<CpuDC::tensor_t> output_minus_delta;
			layer_minus_delta.act(input, output_minus_delta, nullptr);

			const auto weight_grad_reference = (output_plus_delta.sum() - output_minus_delta.sum()) * one_over_double_delta;
			const auto weight_grad_actual = param_container_grad[item_id];
			const auto diff = std::abs(weight_grad_reference - weight_grad_actual);
			max_gradient_diff = std::max(max_gradient_diff, diff);
		}

		StandardTestUtils::LogAndAssertLessOrEqualTo(
			"Deviation between actual and expected gradients with respect to the weight matrix",
			max_gradient_diff, tolerance);
	}

}