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

#include "CppUnitTest.h"
#include <NeuralNet/RMLayer.h>
#include "StandardTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(RMLayerTest)
	{
		/// <summary>
		/// Returns an instance of "input data" initialized according to the given set of parameters.
		/// </summary>
		static MLayerData<CpuDC> get_random(const int size,
			const Index3d& item_size, const InitializationStrategy init_strategy)
		{
			MLayerData<CpuDC> result(size);

			for (auto itemId = 0; itemId < size; ++itemId)
			{
				result[itemId].resize(item_size);
				result[itemId].init(init_strategy);
			}

			return result;
		}

		/// <summary>
		/// Returns collection filled with the given value.
		/// </summary>
		static LazyVector<CpuDC::tensor_t> construct_and_fill(const int size, const Index3d& item_size, const Real val)
		{
			LazyVector<CpuDC::tensor_t> result(size);

			for (auto itemId = 0; itemId < size; ++itemId)
			{
				result[itemId].resize(item_size);
				result[itemId].fill(val);
			}

			return result;
		}

		static constexpr auto _delta = std::is_same_v<Real, double> ? static_cast<Real>(1e-5) : static_cast<Real>(1e-2);
		static constexpr auto _tolerance = std::is_same_v<Real, double> ? static_cast<Real>(6e-10) : static_cast<Real>(3e-4);

		TEST_METHOD(InputGradientTest)
		{
			// Arrange
			constexpr auto rec_depth = 5;
			constexpr auto in_size_plain = 10;
			constexpr auto out_size_plain = 8;
			const Index3d in_item_size(1, 1, in_size_plain);
			const Index3d out_item_size(1, 1, out_size_plain);
			const auto out_grad = construct_and_fill(rec_depth, out_item_size, 1);
			const auto input = get_random(rec_depth, in_item_size, FillRandomUniform);
			auto trace_data = input;
			LazyVector<CpuDC::tensor_t> output;
			const RMLayer<CpuDC> layer(rec_depth, in_size_plain, out_item_size,
				FillRandomNormal, ActivationFunctionId::SIGMOID);
			LazyVector<CpuDC::tensor_t> in_gradient(rec_depth);
			auto layer_grad = layer.allocate_gradient_container();

			// Act
			layer.act(input, output, &trace_data);
			layer.backpropagate(out_grad, output, trace_data.Data, in_gradient,
				layer_grad, true /*evaluate input gradient*/);

			// Assert
			Real max_gradient_diff{};
			constexpr auto one_over_double_delta = static_cast<Real>(0.5) / _delta;

			for (auto depth_id = 0ull; depth_id < in_gradient.size(); ++depth_id)
			{
				const auto& in_grad_for_depth = in_gradient[depth_id];
				Assert::AreEqual<std::size_t>(in_grad_for_depth.size(), in_size_plain, 
					L"Invalid dimension of input gradient vector.");

				for (auto item_id = 0; item_id < in_size_plain; ++item_id)
				{
					auto input_plus_delta = input;
					input_plus_delta[depth_id][item_id] += _delta;

					auto input_minus_delta = input;
					input_minus_delta[depth_id][item_id] -= _delta;

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
				max_gradient_diff, _tolerance);
		}

		TEST_METHOD(BiasesGradientTest)
		{
			constexpr auto rec_depth = 5;
			constexpr auto in_size_plain = 10;
			constexpr auto out_size_plain = 8;
			const Index3d in_item_size(1, 1, in_size_plain);
			const Index3d out_item_size(1, 1, out_size_plain);
			const auto out_grad = construct_and_fill(rec_depth, out_item_size, 1);
			const auto input = get_random(rec_depth, in_item_size, FillRandomUniform);
			const CpuDC::matrix_t in_weights(out_size_plain, in_size_plain, -1, 1);
			const CpuDC::matrix_t rec_weights(out_size_plain, out_size_plain, -1, 1);
			const CpuDC::vector_t biases(out_size_plain, -1, 1);
			auto trace_data = input;
			LazyVector<CpuDC::tensor_t> output;
			const RMLayer<CpuDC> layer(rec_depth, out_item_size, in_weights, rec_weights, biases, ActivationFunctionId::SIGMOID);
			LazyVector<CpuDC::tensor_t> in_gradient{};
			auto layer_grad = layer.allocate_gradient_container();

			// Act
			layer.act(input, output, &trace_data);
			layer.backpropagate(out_grad, output, trace_data.Data, in_gradient,
				layer_grad, false /*evaluate input gradient*/);

			// Assert
			Real max_gradient_diff{};
			constexpr auto one_over_double_delta = static_cast<Real>(0.5) / _delta;

			const auto& biases_grad = layer_grad[0].Biases_grad;
			Assert::AreEqual<std::size_t>(biases_grad.size(), out_size_plain,
				L"Invalid dimension of biases gradient vector.");

			for (auto item_id = 0; item_id < out_size_plain; ++item_id)
			{
				auto biases_plus_delta = biases;
				biases_plus_delta[item_id] += _delta;

				LazyVector<CpuDC::tensor_t> output_plus_delta;
				const RMLayer<CpuDC> layer_plus_delta(rec_depth, out_item_size, in_weights,
					rec_weights, biases_plus_delta, ActivationFunctionId::SIGMOID);
				layer_plus_delta.act(input, output_plus_delta, nullptr);

				auto biases_minus_delta = biases;
				biases_minus_delta[item_id] -= _delta;

				LazyVector<CpuDC::tensor_t> output_minus_delta;
				const RMLayer<CpuDC> layer_minus_delta(rec_depth, out_item_size, in_weights,
					rec_weights, biases_minus_delta, ActivationFunctionId::SIGMOID);
				layer_minus_delta.act(input, output_minus_delta, nullptr);

				const auto biases_grad_reference = (output_plus_delta.sum() - output_minus_delta.sum()) * one_over_double_delta;
				const auto biases_grad_actual = biases_grad[item_id];
				const auto diff = std::abs(biases_grad_reference - biases_grad_actual);
				max_gradient_diff = std::max(max_gradient_diff, diff);
			}

			StandardTestUtils::LogAndAssertLessOrEqualTo(
				"Deviation between actual and expected gradients with respect to the bias vector",
				max_gradient_diff, _tolerance);
		}

		/// <summary>
		/// General method to run gradient tests with respect to weights
		/// </summary>
		static void run_weight_gradient_test(const int weight_id)
		{
			constexpr auto rec_depth = 5;
			constexpr auto in_size_plain = 10;
			constexpr auto out_size_plain = 8;
			const Index3d in_item_size(1, 1, in_size_plain);
			const Index3d out_item_size(1, 1, out_size_plain);
			const auto out_grad = construct_and_fill(rec_depth, out_item_size, 1);
			const auto input = get_random(rec_depth, in_item_size, FillRandomUniform);
			const std::vector weights = 
			{ CpuDC::matrix_t(out_size_plain, in_size_plain, -1, 1),
			CpuDC::matrix_t(out_size_plain, out_size_plain, -1, 1)	};
			const CpuDC::vector_t biases(out_size_plain, -1, 1);
			auto trace_data = input;
			LazyVector<CpuDC::tensor_t> output;
			const RMLayer<CpuDC> layer(rec_depth, out_item_size, weights[0], weights[1], biases, ActivationFunctionId::SIGMOID);
			LazyVector<CpuDC::tensor_t> in_gradient{};
			auto layer_grad = layer.allocate_gradient_container();

			// Act
			layer.act(input, output, &trace_data);
			layer.backpropagate(out_grad, output, trace_data.Data, in_gradient,
				layer_grad, false /*evaluate input gradient*/);

			// Assert
			Real max_gradient_diff{};
			constexpr auto one_over_double_delta = static_cast<Real>(0.5) / _delta;

			const auto& weights_grad = layer_grad[0].Weights_grad[weight_id];
			Assert::IsTrue(weights_grad.size_3d() == weights[weight_id].size_3d(),
				L"Invalid dimension of biases gradient vector.");

			for (auto item_id = 0; item_id < weights_grad.size(); ++item_id)
			{
				auto weights_plus_delta = weights;
				weights_plus_delta[weight_id][item_id] += _delta;

				LazyVector<CpuDC::tensor_t> output_plus_delta;
				const RMLayer<CpuDC> layer_plus_delta(rec_depth, out_item_size, weights_plus_delta[0],
					weights_plus_delta[1], biases, ActivationFunctionId::SIGMOID);
				layer_plus_delta.act(input, output_plus_delta, nullptr);

				auto weights_minus_delta = weights;
				weights_minus_delta[weight_id][item_id] -= _delta;

				LazyVector<CpuDC::tensor_t> output_minus_delta;
				const RMLayer<CpuDC> layer_minus_delta(rec_depth, out_item_size, weights_minus_delta[0],
					weights_minus_delta[1], biases, ActivationFunctionId::SIGMOID);
				layer_minus_delta.act(input, output_minus_delta, nullptr);

				const auto weight_grad_reference = (output_plus_delta.sum() - output_minus_delta.sum()) * one_over_double_delta;
				const auto weight_grad_actual = weights_grad[item_id];
				const auto diff = std::abs(weight_grad_reference - weight_grad_actual);
				max_gradient_diff = std::max(max_gradient_diff, diff);
			}

			StandardTestUtils::LogAndAssertLessOrEqualTo(
				"Deviation between actual and expected gradients with respect to the weight matrix",
				max_gradient_diff, _tolerance);
		}

		TEST_METHOD(InWeightGradientTest)
		{
			run_weight_gradient_test(0);
		}

		TEST_METHOD(RecWeightGradientTest)
		{
			run_weight_gradient_test(1);
		}

		TEST_METHOD(SerializationTest)
		{
			// Arrange
			const Index4d in_size{ {6, 9, 7}, 10 };
			const Index4d out_size{ {8, 11, 5}, 10 };
			const RMLayer<CpuDC> layer(in_size, out_size, FillRandomNormal, ActivationFunctionId::SIGMOID);

			// Act 
			const auto msg = MsgPack::pack(layer);
			const auto layer_unpacked = MsgPack::unpack<RMLayer<CpuDC>>(msg);

			//Assert
			Assert::IsTrue(layer.equal(layer_unpacked), L"Original and restored layers are different");
		}
	};
}