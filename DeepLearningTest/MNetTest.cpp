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
#include <Math/CostFunctionFactory.h>
#include <Math/CollectionArithmetics.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(MNetTest)
	{
		static constexpr auto _delta = std::is_same_v<Real, double> ? static_cast<Real>(1e-5) : static_cast<Real>(1e-2);
		static constexpr auto _tolerance = std::is_same_v<Real, double> ? static_cast<Real>(6e-10) : static_cast<Real>(3e-4);

		/// <summary>
		/// Returns a multi-net consisting of a few layers.
		/// </summary>
		static MNet<CpuDC> build_net(const bool single_layer = false)
		{
			constexpr auto rec_depth = 5;
			constexpr auto in_size_plain = 10;
			Index4d in_size({ 1, 1, in_size_plain }, rec_depth);

			// Construct net with two layers.
			MNet<CpuDC> net{};
			in_size = net.append_layer<RMLayer>(in_size, Index4d{ { 1, 1, 7 }, rec_depth },
				FillRandomNormal, ActivationFunctionId::SIGMOID);

			if (!single_layer)
			{
				in_size = net.append_layer<RMLayer>(in_size, Index4d{ { 1, 1, 11 }, rec_depth },
					FillRandomNormal, ActivationFunctionId::SIGMOID);
				in_size = net.append_layer<RMLayer>(in_size, Index4d{ { 1, 1, 9 }, rec_depth },
					FillRandomNormal, ActivationFunctionId::SIGMOID);
			}

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

		TEST_METHOD(RandomGeneratorTest)
		{
			// Act
			MNet<CpuDC>::reset_random_generator(0);
			const auto net0 = build_net();
			const auto net1 = build_net();
			MNet<CpuDC>::reset_random_generator(0);
			const auto net2 = build_net();
			MNet<CpuDC>::reset_random_generator();

			// Assert
			Assert::IsFalse(net0 == net1, L"The nets are not supposed to be equal");
			Assert::IsTrue(net0 == net2, L"The nets are supposed to be equal");
		}


		TEST_METHOD(GradientSumTest)
		{
			// Arrange
			const auto net = build_net();
			const auto in_size = net.in_size();

			const std::vector input{ MNetTestUtils::construct_random_vector<CpuDC>(in_size.w, in_size.xyz),
			MNetTestUtils::construct_random_vector<CpuDC>(in_size.w, in_size.xyz) };

			const auto out_size = net.out_size();

			const std::vector reference{ MNetTestUtils::construct_random_vector<CpuDC>(out_size.w, out_size.xyz),
			MNetTestUtils::construct_random_vector<CpuDC>(out_size.w, out_size.xyz) };

			const auto cost_function = CostFunction<CpuDC::tensor_t>(CostFunctionId::SQUARED_ERROR);

			// Act
			const auto gradient_sum = net.calc_gradient_sum(input, reference, cost_function);

			// Assert
			const auto gradient_sum_reference =
				net.calc_gradient(input[0], reference[0], cost_function) +
				net.calc_gradient(input[1], reference[1], cost_function);

			Assert::AreEqual(gradient_sum.size(), gradient_sum_reference.size(), L"Gradient containers have different sizes");

			auto diff = static_cast<Real>(0);

			for (auto layer_id = 0ull; layer_id < gradient_sum.size(); ++layer_id)
			{
				Assert::AreEqual(gradient_sum[layer_id].size(), gradient_sum_reference[layer_id].size(),
					L"Gradient sub-containers have different sizes");

				for (auto item_id = 0ull; item_id < gradient_sum[layer_id].size(); ++item_id)
				{
					const auto local_diff = (gradient_sum[layer_id][item_id] - gradient_sum_reference[layer_id][item_id]).max_abs();
					diff = std::max(local_diff, diff);
				}
			}

			Assert::IsTrue(diff < 10 * std::numeric_limits<Real>::epsilon(), L"Gradient sums are not equal");
		}

		/// <summary>
		/// Returns collection of random items that can be used as an input for a multi-net.
		/// </summary>
		static std::vector<LazyVector<CpuDC::tensor_t>> generate_random_input(const int count, const Index4d& item_size)
		{
			std::vector<LazyVector<CpuDC::tensor_t>> result;

			for (auto item_id = 0; item_id < count; ++item_id)
				result.emplace_back(MNetTestUtils::construct_random_vector<CpuDC>(item_size.w, item_size.xyz));

			return result;
		}

		/// <summary>
		/// Evaluates the given <paramref name="net"/> at the given <paramref name="input"/> and returns the result.
		/// </summary>
		static std::vector<LazyVector<CpuDC::tensor_t>> evaluate(const std::vector<LazyVector<CpuDC::tensor_t>>& input,
			const MNet<CpuDC>& net)
		{
			std::vector<LazyVector<CpuDC::tensor_t>> result;

			for (auto item_id = 0ull; item_id < input.size(); ++item_id)
				result.emplace_back(net.act(input[item_id]));

			return result;
		}

		/// <summary>
		/// Returns average (item "0") and maximal (item "1") absolute deviations between the
		/// corresponding items of the two given collections of the same size.
		/// </summary>
		static std::tuple<Real, Real> evaluate_average_diff(const std::vector<LazyVector<CpuDC::tensor_t>>& collection0,
			const std::vector<LazyVector<CpuDC::tensor_t>>& collection1)
		{
			Assert::AreEqual(collection1.size(), collection0.size(),
				L"The collections must be of the same size");

			Real diff_sum{};
			Real max_diff{};

			for (auto item_id = 0ull; item_id < collection0.size(); ++item_id)
			{
				const auto max_abs = (collection0[item_id] - collection1[item_id]).max_abs();
				diff_sum += max_abs;
				max_diff = std::max(max_abs, max_diff);
			}

			return std::make_tuple(diff_sum / collection0.size(), max_diff);
		}

		TEST_METHOD(LearningTest)
		{
			// Arrange
			MNet<CpuDC>::reset_random_generator(0);
			const auto ref_net = build_net(true /*single_layer*/);
			auto net = build_net(true /*single_layer*/);
			const auto in_size = net.in_size();
			const auto cost_function = CostFunction<CpuDC::tensor_t>(CostFunctionId::CROSS_ENTROPY);

			const auto check_input = generate_random_input(100, in_size);
			const auto check_reference = evaluate(check_input, ref_net);

			const auto [init_average_diff, init_max_diff] = evaluate_average_diff(check_reference, evaluate(check_input, net));
			Assert::IsTrue(init_average_diff > static_cast<Real>(0.5), L"Too low initial difference.");

			// Act
			auto learning_rate = static_cast<Real>(4e-1);
			constexpr auto learning_iterations = 10000;
			constexpr auto tolerance = std::is_same_v<Real, double> ? static_cast<Real>(1e-10) : static_cast<Real>(1e-6);

			for (auto i = 0; i < learning_iterations; ++i)
			{
				const auto input = generate_random_input(10, in_size);
				const auto reference = evaluate(input, ref_net);

				net.learn(input, reference, cost_function, learning_rate);
			}

			// Assert
			const auto [final_average_diff, final_max_diff] = evaluate_average_diff(check_reference, evaluate(check_input, net));

			StandardTestUtils::LogAndAssertLessOrEqualTo<double>("Final difference",
				final_max_diff, tolerance);
		}

		TEST_METHOD(SerializationTest)
		{
			//Arrange
			const auto net = build_net();

			//Act
			const auto msg = MsgPack::pack(net);
			const auto net_unpacked = MsgPack::unpack<MNet<CpuDC>>(msg);

			//Assert
			Assert::IsTrue(net == net_unpacked, L"Original and restored nets are different");
		}
	};
}
