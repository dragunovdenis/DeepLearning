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
#include <NeuralNet/MNet.h>
#include "MNetTestUtils.h"
#include "StandardTestUtils.h"
#include "Math/CostFunctionFactory.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(MNetTest)
	{
		static constexpr auto _delta = std::is_same_v<Real, double> ? static_cast<Real>(1e-5) : static_cast<Real>(1e-2);
		static constexpr auto _tolerance = std::is_same_v<Real, double> ? static_cast<Real>(5e-10) : static_cast<Real>(3e-4);

		/// <summary>
		/// Returns a multi-net consisting of a few layers.
		/// </summary>
		static MNet<CpuDC> build_net()
		{
			constexpr auto rec_depth = 5;
			constexpr auto in_size_plain = 10;
			Index4d in_size({ 1, 1, in_size_plain }, rec_depth);

			// Construct net with two layers.
			MNet<CpuDC> net{};
			in_size = net.append_layer<RMLayer>(in_size, Index4d{ { 1, 1, 7 }, rec_depth },
				FillRandomNormal, ActivationFunctionId::SIGMOID);
			in_size = net.append_layer<RMLayer>(in_size, Index4d{ { 1, 1, 11 }, rec_depth },
				FillRandomNormal, ActivationFunctionId::SIGMOID);
			in_size = net.append_layer<RMLayer>(in_size, Index4d{ { 1, 1, 9 }, rec_depth },
				FillRandomNormal, ActivationFunctionId::SIGMOID);

			return net;
		}

		/// <summary>
		/// A general method to run gradient tests for a multi-net.
		/// </summary>
		static void run_general_gradient_test(const int param_container_id)
		{
			// Arrange
			const auto net = build_net();
			const auto in_size = net.in_size();
			const auto input = MNetTestUtils::construct_random_vector<CpuDC>(in_size.w, in_size.xyz);
			const auto out_size = net.out_size();
			const auto reference = MNetTestUtils::construct_random_vector<CpuDC>(out_size.w, out_size.xyz);
			const auto zero_gradients = net.allocate_gradients(true /*fill zero*/);

			// Act 
			const auto cost_function = CostFunction<CpuDC::tensor_t>(CostFunctionId::SQUARED_ERROR);
			const auto gradients = net.calc_gradient(input, reference, cost_function);

			// Assert
			constexpr auto one_over_double_delta = static_cast<Real>(0.5) / _delta;

			for (auto layer_id = 0; layer_id < net.layer_count(); ++layer_id)
			{
				const auto grad_container = gradients[layer_id][0].data[param_container_id];
				auto max_gradient_diff = static_cast<Real>(0);

				for (auto item_id = 0ull; item_id < grad_container.size(); ++item_id)
				{
					auto zero_gradient_plus_delta = zero_gradients;
					zero_gradient_plus_delta[layer_id][0].data[param_container_id][item_id] = _delta;

					auto net_plus_delta = net;
					net_plus_delta.update(zero_gradient_plus_delta, static_cast<Real>(1));
					const auto cost_plus_delta = MNetTestUtils::evaluate_cost<CpuDC>(cost_function, 
						net_plus_delta.act(input), reference);

					auto net_minus_delta = net;
					net_minus_delta.update(zero_gradient_plus_delta, static_cast<Real>(-1));
					const auto cost_minus_delta = MNetTestUtils::evaluate_cost<CpuDC>(cost_function,
						net_minus_delta.act(input), reference);

					const auto grad_reference = (cost_plus_delta - cost_minus_delta) * one_over_double_delta;
					const auto grad_actual = grad_container[item_id];

					const auto diff = std::abs(grad_reference - grad_actual);
					max_gradient_diff = std::max(max_gradient_diff, diff);
				}

				StandardTestUtils::LogAndAssertLessOrEqualTo(
					"Deviation between actual and expected gradients (layer " + std::to_string(layer_id) + ")",
					max_gradient_diff, _tolerance);
			}
		}

		TEST_METHOD(BiasesGradientTest)
		{
			run_general_gradient_test(RMLayer<CpuDC>::BIAS_GRAD_ID);
		}

		TEST_METHOD(InWeightGradientTest)
		{
			run_general_gradient_test(RMLayer<CpuDC>::IN_W_GRAD_ID);
		}

		TEST_METHOD(RecWeightGradientTest)
		{
			run_general_gradient_test(RMLayer<CpuDC>::REC_W_GRAD_ID);
		}
	};
}
