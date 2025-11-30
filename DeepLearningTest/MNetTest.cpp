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
		static constexpr auto _tolerance = std::is_same_v<Real, double> ? static_cast<Real>(6e-10) : static_cast<Real>(4e-4);

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
					net_plus_delta.update(zero_gradient_plus_delta, static_cast<Real>(1), static_cast<Real>(0) /*reg factor*/);
					const auto cost_plus_delta = MNetTestUtils::evaluate_cost<CpuDC>(cost_function, 
						net_plus_delta.act(input), reference);

					auto net_minus_delta = net;
					net_minus_delta.update(zero_gradient_plus_delta, static_cast<Real>(-1), static_cast<Real>(0) /*reg factor*/);
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

		/// <summary>
		/// Returns a span-based representation of the given <paramref name="source"/> vector.
		/// </summary>
		static LazyVector<std::span<const CpuDC::tensor_t>> to_vector_of_spans(const LazyVector<LazyVector<CpuDC::tensor_t>>& source)
		{
			LazyVector<std::span<const CpuDC::tensor_t>> result(source.size());
			std::ranges::transform(source, result.begin(), [](const auto& x) { return x.to_span_read_only(); });

			return result;
		}

		TEST_METHOD(GradientSumTest)
		{
			// Arrange
			const auto net = build_net();
			const auto in_size = net.in_size();

			const LazyVector input_raw{ MNetTestUtils::construct_random_vector<CpuDC>(in_size.w, in_size.xyz),
			MNetTestUtils::construct_random_vector<CpuDC>(in_size.w, in_size.xyz) };
			const auto input = to_vector_of_spans(input_raw);

			const auto out_size = net.out_size();

			const LazyVector reference_raw{ MNetTestUtils::construct_random_vector<CpuDC>(out_size.w, out_size.xyz),
			MNetTestUtils::construct_random_vector<CpuDC>(out_size.w, out_size.xyz) };
			const auto reference = to_vector_of_spans(reference_raw);

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
		static LazyVector<LazyVector<CpuDC::tensor_t>> generate_random_input(const int count, const Index4d& item_size)
		{
			LazyVector<LazyVector<CpuDC::tensor_t>> result(count);

			for (auto item_id = 0; item_id < count; ++item_id)
				result[item_id] = MNetTestUtils::construct_random_vector<CpuDC>(item_size.w, item_size.xyz);

			return result;
		}

		/// <summary>
		/// Evaluates the given <paramref name="net"/> at the given <paramref name="input"/> and returns the result.
		/// </summary>
		static LazyVector<LazyVector<CpuDC::tensor_t>> evaluate(const LazyVector<LazyVector<CpuDC::tensor_t>>& input,
			const MNet<CpuDC>& net)
		{
			LazyVector<LazyVector<CpuDC::tensor_t>> result(input.size());

			for (auto item_id = 0ull; item_id < input.size(); ++item_id)
				result[item_id] = net.act(input[item_id]);

			return result;
		}

		/// <summary>
		/// Returns average (item "0") and maximal (item "1") absolute deviations between the
		/// corresponding items of the two given collections of the same size.
		/// </summary>
		static std::tuple<Real, Real> evaluate_average_diff(const LazyVector<LazyVector<CpuDC::tensor_t>>& collection0,
			const LazyVector<LazyVector<CpuDC::tensor_t>>& collection1)
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
				const auto input_raw = generate_random_input(10, in_size);
				const auto input = to_vector_of_spans(input_raw);
				const auto reference_raw = evaluate(input_raw, ref_net);
				const auto reference = to_vector_of_spans(reference_raw);

				net.learn(input, reference, cost_function, learning_rate, static_cast<Real>(0) /*reg factor*/);
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

		TEST_METHOD(SineSequencePredictionTest)
		{
			Real max_error = 0;
			Real average_error = 0;
			auto iteration_count = 1;
			for (auto i = 0; i < iteration_count; i++)
			{
				const auto error = RunSineSequenceTraining();
				max_error = std::max(error, max_error);
				average_error += error;
			}

			average_error /= iteration_count;

			StandardTestUtils::Log("Average error", average_error);
			StandardTestUtils::Log("Total iterations", iteration_count);
			StandardTestUtils::LogAndAssertLessOrEqualTo<Real>(
				"Maximum prediction error ", max_error, static_cast<Real>(4e-4));
		}

		/// <summary>
		/// Returns a span-based representation of the given <paramref name="source"/>
		/// vector with the last item of each sub-vector being skipped.
		/// </summary>
		static LazyVector<std::span<const CpuDC::tensor_t>> to_vector_of_input_spans(const LazyVector<LazyVector<CpuDC::tensor_t>>& source)
		{
			LazyVector<std::span<const CpuDC::tensor_t>> result(source.size());
			std::ranges::transform(source, result.begin(), [](const auto& x)
				{ return x.to_span_read_only(0, 1); });

			return result;
		}

		/// <summary>
		/// Returns a span-based representation of the given <paramref name="source"/>
		/// vector with the first item of each sub-vector being skipped.
		/// </summary>
		static LazyVector<std::span<const CpuDC::tensor_t>> to_vector_of_target_spans(const LazyVector<LazyVector<CpuDC::tensor_t>>& source)
		{
			LazyVector<std::span<const CpuDC::tensor_t>> result(source.size());
			std::ranges::transform(source, result.begin(), [](const auto& x)
				{ return x.to_span_read_only(1, 0); });

			return result;
		}


		/// <summary>
		/// Trains the given neural net according to the given set of parameters.
		/// </summary>
		static void train(MNet<CpuDC>& net, const int depth, const int batch_size, const Real delta_t,
			const int learning_iterations, const Real step_begin, const Real step_end,
			const std::function<LazyVector<LazyVector<CpuDC::tensor_t>>(int, int)>& allocator,
			const std::function<void(Real, LazyVector<CpuDC::tensor_t>&, int)>& sequence_generator)
		{
			constexpr auto two_pi = static_cast<Real>(3.141592 * 2);
			const auto batch_sampling_step = two_pi / batch_size;
			const auto cost_function = CostFunction<CpuDC::tensor_t>(CostFunctionId::SQUARED_ERROR);

			// Train the network
			// Input: a sequence of sine values
			// Output: a sine sequence shifted one step forward with respect to the input sequence
			auto batch = allocator(depth + 1, batch_size);

			for (auto iter = 0; iter < learning_iterations; ++iter)
			{
				// Learning rate decay: linearly decrease from initial to final
				const Real progress = static_cast<Real>(iter) / static_cast<Real>(learning_iterations);
				Real learning_rate = step_begin * (static_cast<Real>(1) - progress) + step_end * progress;

				learning_rate *= learning_rate;
				learning_rate *= learning_rate;

				auto t_start = Utils::get_random(static_cast<Real>(0), two_pi);

				// Generate random training sequences
				for (auto b = 0; b < batch_size; ++b)
				{
					sequence_generator(t_start, batch[b], depth + 1);
					t_start += batch_sampling_step;
				}

				const auto input = to_vector_of_input_spans(batch);
				const auto target = to_vector_of_target_spans(batch);

				net.learn(input, target, cost_function,
				          learning_rate, static_cast<Real>(0.1));
			}
		}

		/// <summary>
		/// Returns maximal prediction error after training a deep RNN on a
		/// sequences of equidistantly samples values of sine function.
		/// </summary>
		Real RunSineSequenceTraining() const
		{
			// Arrange: Build the network, allocate data structures
			constexpr auto delta_t = static_cast<Real>(1.0/3.0);  // Sample interval (smaller = smoother)
			constexpr auto gradient_clip_threshold = static_cast<Real>(5);
			constexpr auto weight_scale_factor = static_cast<Real>(0.7);
			constexpr auto batch_size = 7;

			MNet<CpuDC> net{};
			Index4d in_size({ 1, 1, 1 }, -1);  // Single value per timestep
			in_size = net.append_layer<RMLayer>(in_size, Index4d{ { 1, 1, 6 }, -1 },
				FillRandomNormal, ActivationFunctionId::LINEAR, gradient_clip_threshold, weight_scale_factor);
			in_size = net.append_layer<RMLayer>(in_size, Index4d{ { 1, 1, 1 }, -1 },
				FillRandomNormal, ActivationFunctionId::LINEAR, gradient_clip_threshold, weight_scale_factor);

			auto func = [](const auto x) -> Real { return static_cast<Real>(std::sin(x)); };

			// Lambda-function to generate sine sequence starting at the given  time "start_t"
			auto generate_time_sequence = [&](const Real t_start, LazyVector<CpuDC::tensor_t>& out_result, const int d)
				{
					for (auto i = 0; i < d; ++i)
						out_result[i](0, 0, 0) = func(t_start + i * delta_t);
				};

			// Lambda-function to allocate input and output containers
			auto allocate_lazy_container = [=](const int d_size, const int b_size) -> LazyVector<LazyVector<CpuDC::tensor_t>>
				{
					LazyVector<LazyVector<CpuDC::tensor_t>> result(b_size);

					for (auto batch_item_id = 0; batch_item_id < b_size; ++batch_item_id)
					{
						result[batch_item_id].resize(d_size);
						auto& item = result[batch_item_id];
						for (auto i = 0; i < d_size; ++i)
							item[i].resize({ 1, 1, 1 });
					}

					return result;
				};

			// Train
			int local_depth;

			for (local_depth = 5; local_depth < 60; local_depth += 5)
				train(net, local_depth, batch_size, delta_t, 50,
					static_cast<Real>(0.84), static_cast<Real>(0.84),
					allocate_lazy_container, generate_time_sequence);

			train(net, local_depth, batch_size, delta_t, 2000,
				static_cast<Real>(0.84), static_cast<Real>(0.1),
				allocate_lazy_container, generate_time_sequence);

			// Evaluate approximation error at a predicted value
			auto test_time = static_cast<Real>(1);
			auto test_count = 20;
			auto error_sum = static_cast<Real>(0);
			auto input_data = allocate_lazy_container(local_depth, 1)[0];

			for (auto test_id = 0; test_id < test_count; test_id++)
			{
				test_time += test_id * static_cast<Real>(0.1);
				auto test_start_time = test_time - local_depth * delta_t;
				generate_time_sequence(test_start_time, input_data, local_depth);

				const auto prediction = net.act(input_data);
				const auto predicted_value = prediction[local_depth - 1](0, 0, 0);

				const auto error = std::abs(predicted_value - func(test_time));
				error_sum += error;
			}

			auto average_error = error_sum / test_count;

			return average_error;
		}

		TEST_METHOD_CLEANUP(CleanupCheck)
		{
			const auto alive_instances = BasicCollection::get_total_instances_count();
			const auto occupied_memory = BasicCollection::get_total_allocated_memory();

			Logger::WriteMessage((std::to_string(alive_instances) + " alive instance(-s) of `BasicCollection` occupying "
				+ std::to_string(occupied_memory) + " byte(-s) of memory.\n").c_str());
		}
	};
}
