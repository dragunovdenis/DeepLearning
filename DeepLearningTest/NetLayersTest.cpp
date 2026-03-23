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

#include "CppUnitTest.h"
#include <NeuralNet/NLayer.h>
#include <NeuralNet/CLayer.h>
#include <NeuralNet/LayerHandle.h>
#include <Math/CostFunction.h>
#include <Utilities.h>
#include <MsgPackUtils.h>
#include <optional>
#include <numeric>
#include <algorithm>
#include <random>
#include "StandardTestUtils.h"
#include "NetLayersTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NetLayersTest)
	{

		/// <summary>
		/// Returns a "standard" CLayer instance to be used in testing
		/// </summary>
		static CLayer<CpuDC> CreateCpuCLayer(const ActivationFunctionId activation = ActivationFunctionId::SIGMOID)
		{
			return CreateCLayer<CpuDC>(activation);
		}

		/// <summary>
		/// Returns a "standard" PLayer instance to be used in testing
		/// </summary>
		static PLayer<CpuDC> CreateCpuPLayer(const PoolTypeId pool_oper_id = PoolTypeId::MAX)
		{
			return CreatePLayer<CpuDC>(pool_oper_id);
		}

		/// <summary>
		/// Returns a "standard" NLayer instance to be used in testing
		/// </summary>
		static NLayer<CpuDC> CreateCpuNLayer(const ActivationFunctionId activation_func_id = ActivationFunctionId::SIGMOID)
		{
			return CreateNLayer<CpuDC>(activation_func_id);
		}

		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to the input 
		/// </summary>
		static void RunGeneralDerivativeWithRespectToInputValuesTest(const ALayer<CpuDC>& nl, const CostFunctionId cost_function_id, const Real& tolerance,
																			 const Real& delta = std::is_same_v<Real, double> ? Real(1e-5) : Real(1e-3),
																			 const std::optional<Tensor>& input_op = std::nullopt)
		{
			const auto input = input_op.has_value() ? input_op.value() : Tensor(nl.in_size(), -1, 1);
			const auto reference = Tensor(nl.out_size(), -1, 1);;

			const auto cost_func = CostFunction<Tensor>(cost_function_id);
			auto layer_data = LayerData<CpuDC>(input);
			const auto [value, cost_gradient] = cost_func.func_and_deriv(nl.
				act(layer_data.Input, &layer_data.Trace), reference);
			const auto [input_grad_result, layer_grad_result] = nl.backpropagate(cost_gradient, layer_data);

			//Assert
			Assert::IsTrue(input_grad_result.size_3d() == input.size_3d(), L"Unexpected size of the gradient with respect to the input tensor");

			auto max_diff = Real(0);
			for (std::size_t in_item_id = 0; in_item_id < input.size(); in_item_id++)
			{
				//Calculate partial derivative with respect to the corresponding input item numerically
				auto input_minus_delta = input;
				input_minus_delta[in_item_id] -= delta;
				auto input_plus_delta = input;
				input_plus_delta[in_item_id] += delta;

				const auto result_minus = cost_func(nl.act(input_minus_delta), reference);
				const auto result_plus = cost_func(nl.act(input_plus_delta), reference);
				const auto deriv_numeric = (result_plus - result_minus) / (2 * delta);

				//Now do the same using the back-propagation approach
				const auto diff = std::abs(deriv_numeric - input_grad_result[in_item_id]);
				Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + '\n').c_str());
				max_diff = std::max(max_diff, diff);
				Assert::IsTrue(diff <= tolerance, L"Unexpectedly high deviation!");
			}
			Logger::WriteMessage((std::string("Max. Difference = ") + Utils::to_string(max_diff) + '\n').c_str());
		}

		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to weights and biases
		/// </summary>
		template <class L>
		void RunGeneralDerivativeWithRespectToWeightsAndBiasesTest( const L& nl, const CostFunctionId cost_function_id,
			const Real& tolerance_weights, const Real& tolerance_biases)
		{
			const auto input = Tensor(nl.in_size(), static_cast<Real>(-1), static_cast<Real>(1));
			const auto weight_tensor_size = nl.weight_tensor_size();
			const auto out_size = nl.out_size();
			const auto filters_count = out_size.x;

			const auto reference = Tensor(out_size, -1, 1);
			const auto cost_func = CostFunction<Tensor>(cost_function_id);

			//Act
			auto layer_data = LayerData<CpuDC>(input);
			const auto [value, cost_gradient] = cost_func.func_and_deriv(nl.
				act(layer_data.Input, &layer_data.Trace), reference);
			const auto [input_grad_result, layer_grad_result] = nl.backpropagate(cost_gradient, layer_data);
			const auto grad_data = layer_grad_result.data;

			//Assert
			Assert::IsTrue(grad_data.size() == filters_count + 1 && std::all_of(grad_data.begin() + 1, grad_data.end(),
				[&](const auto& x) { return x.size_3d() == weight_tensor_size; }), L"Unexpected size of the weights gradient data structure");

			Assert::IsTrue(grad_data[0].size_3d() == out_size, L"Unexpected size of the biases gradient data structure");

			LayerGradient<CpuDC> zero_gradient;
			nl.allocate(zero_gradient, /*fill zero*/ true);
			const auto zero_grad_data = zero_gradient.data;

			constexpr auto delta = std::is_same_v<Real, double> ? static_cast<Real>(1e-5) : static_cast<Real>(1e-3);

			auto weights_max_diff = static_cast<Real>(0);
			//Check derivatives with respect to weights
			Logger::WriteMessage("Weights:\n");
			for (auto filter_id = 0ll; filter_id < filters_count; filter_id++)
				for (auto layer_id = 0ll; layer_id < weight_tensor_size.x; layer_id++)
					for (auto row_id = 0ll; row_id < weight_tensor_size.y; row_id++)
						for (auto col_id = 0ll; col_id < weight_tensor_size.z; col_id++)
						{
							//Calculate partial derivative with respect to the corresponding weight
							auto weights_minus_delta = zero_grad_data;
							weights_minus_delta[filter_id + 1](layer_id, row_id, col_id) = -delta;

							auto layer_minus = nl;//create a mutable copy of the initial layer
							layer_minus.update(LayerGradient<CpuDC>{weights_minus_delta }, 1, 0);
							const auto result_minus = cost_func(layer_minus.act(input), reference);

							auto layer_plus = nl;//create a mutable copy of the initial layer
							layer_plus.update(LayerGradient<CpuDC>{ weights_minus_delta }, -1, 0);
							const auto result_plus = cost_func(layer_plus.act(input), reference);

							const auto deriv_numeric = (result_plus - result_minus) / (2 * delta);

							//Now do the same using the back-propagation approach
							const auto diff = std::abs(deriv_numeric - grad_data[filter_id + 1](layer_id, row_id, col_id));
							Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + '\n').c_str());
							weights_max_diff = std::max(weights_max_diff, diff);
							Assert::IsTrue(diff <= tolerance_weights, L"Unexpectedly high deviation (weight derivatives)!");
						}

			Logger::WriteMessage((std::string("Max. Difference = ") + Utils::to_string(weights_max_diff) + '\n').c_str());

			auto biases_max_diff = static_cast<Real>(0);
			//Check derivatives with respect to biases
			Logger::WriteMessage("Biases:\n");
			for (auto layer_id = 0ll; layer_id < out_size.x; layer_id++)
				for (auto row_id = 0ll; row_id < out_size.y; row_id++)
					for (auto col_id = 0ll; col_id < out_size.z; col_id++)
					{
						//Calculate partial derivative with respect to the corresponding bias
						auto biases_minus_delta = zero_grad_data;
						biases_minus_delta[0](layer_id, row_id, col_id) = -delta;

						auto layer_minus = nl;//create a mutable copy of the initial layer
						layer_minus.update(LayerGradient<CpuDC>{biases_minus_delta }, 1, 0);
						const auto result_minus = cost_func(layer_minus.act(input), reference);

						auto layer_plus = nl;//create a mutable copy of the initial layer
						layer_plus.update(LayerGradient<CpuDC>{ biases_minus_delta }, -1, 0);
						const auto result_plus = cost_func(layer_plus.act(input), reference);

						const auto deriv_numeric = (result_plus - result_minus) / (2 * delta);

						//Now do the same using the back-propagation approach
						const auto diff = std::abs(deriv_numeric - grad_data[0](layer_id, row_id, col_id));
						Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + '\n').c_str());
						biases_max_diff = std::max(biases_max_diff, diff);
						Assert::IsTrue(diff <= tolerance_biases, L"Unexpectedly high deviation (bias derivatives)!");
					}

			Logger::WriteMessage((std::string("Max. Difference = ") + Utils::to_string(biases_max_diff) + '\n').c_str());
		}

		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to the input
		/// </summary>
		static void CheckDerivativeWithRespectToInputValuesCalculation(const ActivationFunctionId activation_func_id, const CostFunctionId cost_function_id)
		{
			const auto nl = CreateCpuNLayer(activation_func_id);
			RunGeneralDerivativeWithRespectToInputValuesTest(nl, cost_function_id, (std::is_same_v<Real, double> ? Real(2e-9) : Real(7.0e-3)));
		}

		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to weights and biases
		/// </summary>
		void CheckDerivativeWithRespectToWeightsAndBiasesCalculation(const ActivationFunctionId activation_func_id, const CostFunctionId cost_function_id)
		{
			const auto nl = CreateCpuNLayer(activation_func_id);
			RunGeneralDerivativeWithRespectToWeightsAndBiasesTest(nl, cost_function_id,
				(std::is_same_v<Real, double> ? Real(8e-10) : Real(3.5e-3)),
				(std::is_same_v<Real, double> ? Real(7e-10) : Real(3e-3)));
		}

		TEST_METHOD(NLayerDerivativeWithRespectToInputValuesCalculationSquaredErrorTest)
		{
			CheckDerivativeWithRespectToInputValuesCalculation(ActivationFunctionId::SIGMOID, CostFunctionId::SQUARED_ERROR);
		}

		TEST_METHOD(NLayerDerivativeWithRespectToInputValuesCalculationCrossEntropyTest)
		{
			CheckDerivativeWithRespectToInputValuesCalculation(ActivationFunctionId::SIGMOID, CostFunctionId::CROSS_ENTROPY);
		}

		TEST_METHOD(NLayerDerivativeWithRespectToWeightsAndbiasesCalculationSquaredErrorTest)
		{
			CheckDerivativeWithRespectToWeightsAndBiasesCalculation(ActivationFunctionId::SIGMOID, CostFunctionId::SQUARED_ERROR);
		}

		TEST_METHOD(NLayerDerivativeWithRespectToWeightsAndBiasesCalculationCrossEntropyTest)
		{
			CheckDerivativeWithRespectToWeightsAndBiasesCalculation(ActivationFunctionId::SIGMOID, CostFunctionId::CROSS_ENTROPY);
		}

		TEST_METHOD(NLayerGradientWithScalingTest)
		{
			RunGeneralGradientWithScalingTest(CreateCpuNLayer());
		}

		TEST_METHOD(CLayerGradientWithScalingTest)
		{
			RunGeneralGradientWithScalingTest(CreateCpuCLayer());
		}

		TEST_METHOD(CLayerDerivativeWithRespectToInputValuesCalculationSquaredErrorTest)
		{
			const auto nl = CreateCpuCLayer();

			RunGeneralDerivativeWithRespectToInputValuesTest(nl, CostFunctionId::SQUARED_ERROR,
				(std::is_same_v<Real, double> ? static_cast<Real>(2e-8) : static_cast<Real>(8e-2)));
		}

		TEST_METHOD(CLayerDerivativeWithRespectToWeightsAndBiasesCalculationSquaredErrorTest)
		{
			const auto nl = CreateCpuCLayer();

			RunGeneralDerivativeWithRespectToWeightsAndBiasesTest(nl, CostFunctionId::SQUARED_ERROR,
				(std::is_same_v<Real, double> ? static_cast<Real>(3.1e-8) : static_cast<Real>(2e-1)),
				(std::is_same_v<Real, double> ? static_cast<Real>(6e-9) : static_cast<Real>(3.5e-2)));
		}

		TEST_METHOD(PLayerDerivativeWithRespectToInputValuesCalculationSquaredErrorTest)
		{
			const auto nl = CreateCpuPLayer();
			auto input = Tensor(nl.in_size(), false);
			auto input_vals = std::vector<int>(input.size());
			std::iota(input_vals.begin(), input_vals.end(), 0);

			std::random_device rd;
			std::mt19937 g(rd());
			std::ranges::shuffle(input_vals, g);
			const auto factor = static_cast<Real>(1) / input_vals.size();
			//Fill the input tensor with all the different values with the minimal difference equal to 'factor'
			//This is done in order to avoid ambiguities for the max-pulling operator 
			std::ranges::transform(input_vals, input.begin(), [=](const auto val) { return val * factor; });

			//Set "delta" parameter that will be used in the numerical differentiation procedure
			//so that it is less than the minimal difference between the items in the input tensor
			//(again, to avoid confusion when doing max-pooling)
			RunGeneralDerivativeWithRespectToInputValuesTest(nl, CostFunctionId::SQUARED_ERROR,
				(std::is_same_v<Real, double> ? static_cast<Real>(1e-10) : static_cast<Real>(7e-4)),
				factor/2, std::make_optional<Tensor>(input));
		}

		TEST_METHOD(PLayerDerivativeWithRespectToWeightsAndbiasesCalculationSquaredErrorTest)
		{
			//Arrange
			const auto nl = CreateCpuPLayer();
			const auto input = Tensor(nl.in_size(), static_cast<Real>(-1), static_cast<Real>(1));
			const auto reference = Tensor(nl.out_size(), -1, 1);;

			//Act
			const auto cost_func = CostFunction<Tensor>(CostFunctionId::SQUARED_ERROR);
			auto layer_data = LayerData<CpuDC>(input);
			const auto [value, cost_gradient] = cost_func.func_and_deriv(nl.
				act(layer_data.Input, &layer_data.Trace), reference);
			const auto [input_grad_result, layer_grad_result] = nl.backpropagate(cost_gradient, layer_data);

			//Assert
			Assert::IsTrue(layer_grad_result.data.size() == 0,
				L"Unexpected output of the back-propagation procedure");
		}


		/// <summary>
		/// General method to run layer handle serialization test
		/// </summary>
		template <class L>
		static void RunSerializationTest(const L& layer)
		{
			//Act
			const auto layer_handle = LayerHandle<CpuDC>::template make<L>(layer);
			const auto msg = MsgPack::pack(layer_handle);
			const auto layer_handle_unpacked = MsgPack::unpack<LayerHandle<CpuDC>>(msg);

			//Assert
			const auto tests_samples_count = 100;
			for (int test_sample_id = 0; test_sample_id < tests_samples_count; test_sample_id++)
			{
				//take a random input sample
				const auto input_sample = Tensor(layer.in_size(), -1, 1);
				const auto ref_output = layer.act(input_sample);
				//Sanity check
				Assert::IsTrue(ref_output.max_abs() > 0 && input_sample.max_abs() > 0, L"Both input sample and reference output are expected to be non-zero.");

				const auto trial_output = layer_handle_unpacked.layer().act(input_sample);

				Assert::IsTrue(ref_output == trial_output, L"Layers are not the same.");
			}
		}

		TEST_METHOD(PLayerSerializationTest)
		{
			RunSerializationTest(CreateCpuPLayer());
		}

		TEST_METHOD(CLayerSerializationTest)
		{
			RunSerializationTest(CreateCpuCLayer());
		}

		TEST_METHOD(NLayerSerializationTest)
		{
			RunSerializationTest(NLayer<CpuDC>(10, 23, ActivationFunctionId::SIGMOID));
		}

		/// <summary>
		///	General functional test for the linear activation function
		/// </summary>
		template <class D>
		void test_linear_activation(const ALayer<D>& layer)
		{
			//Arrange
			if (layer.in_size() != Index3d{ 1, 1, 1 } || layer.out_size() != Index3d{ 1, 1, 1 })
				throw std::exception("Inappropriate input/output dimensions of the layer");

			const auto x0 = Utils::get_random(-1, 1);
			const auto x1 = x0 + 10;
			const auto x2 = x1 + 5;
			typename D::tensor_t input(1, 1, 1);

			//Act
			input(0, 0, 0) = x0;
			const auto y0 = layer.act(input)(0, 0, 0);

			input(0, 0, 0) = x1;
			const auto y1 = layer.act(input)(0, 0, 0);

			input(0, 0, 0) = x2;
			const auto y2 = layer.act(input)(0, 0, 0);

			//Assert
			const auto delta0 = (y1 - y0) / (x1 - x0);
			const auto delta1 = (y2 - y1) / (x2 - x1);
			const auto delta2 = (y2 - y0) / (x2 - x0);

			Assert::IsTrue(std::abs(delta0 - delta1) < 10 * std::numeric_limits<Real>::epsilon() &&
				std::abs(delta0 - delta2) < 10 * std::numeric_limits<Real>::epsilon(), 
				L"Too high deviation between the values");
		}

		TEST_METHOD(LinearActivationNLayerTest)
		{
			const auto layer = NLayer(1, 1, ActivationFunctionId::LINEAR);
			test_linear_activation(layer);
		}

		TEST_METHOD(LinearActivationCLayerTest)
		{
			const auto layer = CLayer(Index3d{ 1, 1, 1 },
				Index2d{ 1 , 1}, 1, ActivationFunctionId::LINEAR);
			test_linear_activation(layer);
		}

		/// <summary>
		/// Runs test of the layer "reset" functionality for the given instance of neural net layer;
		/// Important: the layer is supposed to have "linear" activation function
		/// </summary>
		template <class D>
		void ran_general_layer_reset_test(ALayer<D>& layer)
		{
			//Arrange
			const typename D::tensor_t input(layer.in_size(), 1, 2);
			Assert::IsTrue(input.max_abs() > 0, L"Input is not supposed to be zero");
			const typename D::tensor_t zero_input(layer.in_size(), /*assign zero*/ true);
			Assert::IsTrue(zero_input.max_abs() <=0, L"Zero input is supposed to be zero");

			//Sanity check that the layer is not in the "reset" state already
			Assert::IsTrue(layer.squared_weights_sum() > 0, L"Weights are already zero");
			//In the test below we actually use the assumption about "linear" activation function
			Assert::IsTrue(layer.act(zero_input).max_abs() > 0, L"Biases are already zero");

			//Act
			layer.reset();

			//Assert
			Assert::IsTrue(layer.squared_weights_sum() <= 0, L"Weights are are still non-zero");
			Assert::IsTrue(layer.act(input).max_abs() <= 0, L"Biases are are still non-zero");
		}

		TEST_METHOD(CLayerResetTest)
		{
			auto layer = CLayer(Index3d{ 11, 25, 13 },
				Index2d{ 3 , 5 }, 3, ActivationFunctionId::LINEAR);

			ran_general_layer_reset_test(layer);
		}

		TEST_METHOD(NLayerResetTest)
		{
			auto layer = NLayer(35, 123, ActivationFunctionId::LINEAR);
			ran_general_layer_reset_test(layer);
		}
	};
}