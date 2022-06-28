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
#include <type_traits>
#include <optional>
#include <numeric>
#include <algorithm>
#include <random>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NetLayersTest)
	{

		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to the input 
		/// </summary>
		void RunGeneralDerivativeWithRespectToInputValuesTest(const ALayer& nl, const CostFunctionId cost_function_id, const Real& tolerance,
			const Real& delta = std::is_same_v<Real, double> ? Real(1e-5) : Real(1e-3),
			const std::optional<Tensor>& input_op = std::nullopt)
		{
			const auto input = input_op.has_value() ? input_op.value() : Tensor(nl.in_size(), -1, 1);
			const auto reference = Tensor(nl.out_size(), -1, 1);;

			const auto cost_func = CostFunction(cost_function_id);
			auto aux_data = ALayer::AuxLearningData();
			const auto [value, cost_gradient] = cost_func.func_and_deriv(nl.act(input, &aux_data), reference);
			const auto [input_grad_result, layer_grad_result] = nl.backpropagate(cost_gradient, aux_data);

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
			const auto input = Tensor(nl.in_size(), Real(-1), Real(1));
			const auto weight_tensor_size = nl.weight_tensor_size();
			const auto out_size = nl.out_size();
			const auto filters_count = out_size.x;

			const auto reference = Tensor(out_size, -1, 1);;
			const auto cost_func = CostFunction(cost_function_id);

			//Act
			auto aux_data = ALayer::AuxLearningData();
			const auto [value, cost_gradient] = cost_func.func_and_deriv(nl.act(input, &aux_data), reference);
			const auto [input_grad_result, layer_grad_result] = nl.backpropagate(cost_gradient, aux_data);
			const auto wight_grad = layer_grad_result.Weights_grad;
			const auto bias_grad = layer_grad_result.Biases_grad;

			//Assert
			Assert::IsTrue(wight_grad.size() == filters_count && std::all_of(wight_grad.begin(), wight_grad.end(),
				[&](const auto& x) { return x.size_3d() == weight_tensor_size; }), L"Unexpected size of the weights gradient data structure");

			Assert::IsTrue(bias_grad.size_3d() == out_size, L"Unexpected size of the biases gradient data structure");

			const auto zero_weights = std::vector<Tensor>(filters_count, Tensor(weight_tensor_size, true));
			const auto zero_biases = Tensor(out_size, true);

			const auto delta = std::is_same_v<Real, double> ? Real(1e-5) : Real(1e-3);

			auto weights_max_diff = Real(0);
			//Check derivatives with respect to weights
			Logger::WriteMessage("Weights:\n");
			for (auto filter_id = 0ll; filter_id < filters_count; filter_id++)
				for (auto layer_id = 0ll; layer_id < weight_tensor_size.x; layer_id++)
					for (auto row_id = 0ll; row_id < weight_tensor_size.y; row_id++)
						for (auto col_id = 0ll; col_id < weight_tensor_size.z; col_id++)
						{
							//Calculate partial derivative with respect to the corresponding weight
							auto weights_minus_delta = zero_weights;
							weights_minus_delta[filter_id](layer_id, row_id, col_id) = -delta;

							auto layer_minus = nl;//create a mutable copy of the initial layer
							layer_minus.update(std::make_tuple(weights_minus_delta, zero_biases), Real(0));
							const auto result_minus = cost_func(layer_minus.act(input), reference);

							auto weights_plus_delta = zero_weights;
							weights_plus_delta[filter_id](layer_id, row_id, col_id) = delta;

							auto layer_plus = nl;//create a mutable copy of the initial layer
							layer_plus.update(std::make_tuple(weights_plus_delta, zero_biases), Real(0));
							const auto result_plus = cost_func(layer_plus.act(input), reference);

							const auto deriv_numeric = (result_plus - result_minus) / (2 * delta);

							//Now do the same using the back-propagation approach
							const auto diff = std::abs(deriv_numeric - wight_grad[filter_id](layer_id, row_id, col_id));
							Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + '\n').c_str());
							weights_max_diff = std::max(weights_max_diff, diff);
							Assert::IsTrue(diff <= tolerance_weights, L"Unexpectedly high deviation (weight derivatives)!");
						}

			Logger::WriteMessage((std::string("Max. Difference = ") + Utils::to_string(weights_max_diff) + '\n').c_str());

			auto biases_max_diff = Real(0);
			//Check derivatives with respect to biases
			Logger::WriteMessage("Biases:\n");
			for (auto layer_id = 0ll; layer_id < out_size.x; layer_id++)
				for (auto row_id = 0ll; row_id < out_size.y; row_id++)
					for (auto col_id = 0ll; col_id < out_size.z; col_id++)
					{
						//Calculate partial derivative with respect to the corresponding bias
						auto biases_minus_delta = zero_biases;
						biases_minus_delta(layer_id, row_id, col_id) = -delta;

						auto layer_minus = nl;//create a mutable copy of the initial layer
						layer_minus.update(std::make_tuple(zero_weights, biases_minus_delta), Real(0));
						const auto result_minus = cost_func(layer_minus.act(input), reference);

						auto biases_plus_delta = zero_biases;
						biases_plus_delta(layer_id, row_id, col_id) = delta;

						auto layer_plus = nl;//create a mutable copy of the initial layer
						layer_plus.update(std::make_tuple(zero_weights, biases_plus_delta), Real(0));
						const auto result_plus = cost_func(layer_plus.act(input), reference);

						const auto deriv_numeric = (result_plus - result_minus) / (2 * delta);

						//Now do the same using the back-propagation approach
						const auto diff = std::abs(deriv_numeric - bias_grad(layer_id, row_id, col_id));
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
		void CheckDerivativeWithRespectToInputValuesCalculation(const ActivationFunctionId activation_func_id, const CostFunctionId cost_function_id)
		{
			const auto input_dim = 10;
			const auto output_dim = 23;
			const auto nl = NLayer(input_dim, output_dim, activation_func_id);

			RunGeneralDerivativeWithRespectToInputValuesTest(nl, cost_function_id, (std::is_same_v<Real, double> ? Real(2e-9) : Real(5e-3)));
		}

		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to weights and biases
		/// </summary>
		void CheckDerivativeWithRespectToWeightsAndBiasesCalculation(const ActivationFunctionId activation_func_id, const CostFunctionId cost_function_id)
		{
			const auto input_dim = 10;
			const auto output_dim = 23;
			const auto nl = NLayer(input_dim, output_dim, activation_func_id);

			RunGeneralDerivativeWithRespectToWeightsAndBiasesTest(nl, cost_function_id,
				(std::is_same_v<Real, double> ? Real(8e-10) : Real(3e-3)),
				(std::is_same_v<Real, double> ? Real(5e-10) : Real(3e-3)));
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

		/// <summary>
		/// Returns a "standard" CLayer instance to be used in testing
		/// </summary>
		static CLayer ConstructStandardCLayer()
		{
			const auto input_dim = Index3d(5, 13, 17);
			const auto filter_window = Index2d(3);
			const auto filters_count = 7;
			const auto paddings = Index3d(0, 3, 6);
			const auto strides = Index3d(2);
			return CLayer(input_dim, filter_window, filters_count, ActivationFunctionId::SIGMOID, paddings, strides);
		}

		/// <summary>
		/// Returns a "standard" PLayer instance to be used in testing
		/// </summary>
		static PLayer ConstructStandardPLayer()
		{
			const auto input_dim = Index3d(5, 10, 7);
			const auto filter_window = Index2d(3);
			return PLayer(input_dim, filter_window, PoolTypeId::MAX);
		}

		TEST_METHOD(CLayerDerivativeWithRespectToInputValuesCalculationSquaredErrorTest)
		{
			const auto nl = ConstructStandardCLayer();

			RunGeneralDerivativeWithRespectToInputValuesTest(nl, CostFunctionId::SQUARED_ERROR, (std::is_same_v<Real, double> ? Real(2e-8) : Real(8e-2)));
		}

		TEST_METHOD(CLayerDerivativeWithRespectToWeightsAndbiasesCalculationSquaredErrorTest)
		{
			const auto nl = ConstructStandardCLayer();

			RunGeneralDerivativeWithRespectToWeightsAndBiasesTest(nl, CostFunctionId::SQUARED_ERROR,
				(std::is_same_v<Real, double> ? Real(3e-8) : Real(2e-1)),
				(std::is_same_v<Real, double> ? Real(6e-9) : Real(3.5e-2)));
		}

		TEST_METHOD(PLayerDerivativeWithRespectToInputValuesCalculationSquaredErrorTest)
		{
			const auto nl = ConstructStandardPLayer();
			auto input = Tensor(nl.in_size(), false);
			auto input_vals = std::vector<int>(input.size());
			std::iota(input_vals.begin(), input_vals.end(), 0);

			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(input_vals.begin(), input_vals.end(), g);
			const auto factor = Real(1) / input_vals.size();
			//Fill the input tensor with all the different values with the minimal difference equal to 'factor'
			//This is done in order to avoid ambiguities for the max-pulling operator 
			std::transform(input_vals.begin(), input_vals.end(), input.begin(), [=](const auto val) { return val * factor; });

			//Set "delta" parameter that will be used in the numerical differentiation procedure
			//so that it is less than the minimal difference between the items in the input tensor
			//(again, to avoid confusion when doing max-pooling)
			RunGeneralDerivativeWithRespectToInputValuesTest(nl, CostFunctionId::SQUARED_ERROR,
				(std::is_same_v<Real, double> ? Real(1e-10) : Real(4e-4)), factor/2, std::make_optional<Tensor>(input));
		}

		TEST_METHOD(PLayerDerivativeWithRespectToWeightsAndbiasesCalculationSquaredErrorTest)
		{
			//Arrange
			const auto nl = ConstructStandardPLayer();
			const auto input = Tensor(nl.in_size(), Real(-1), Real(1));
			const auto reference = Tensor(nl.out_size(), -1, 1);;

			//Act
			const auto cost_func = CostFunction(CostFunctionId::SQUARED_ERROR);
			auto aux_data = NLayer::AuxLearningData();
			const auto [value, cost_gradient] = cost_func.func_and_deriv(nl.act(input, &aux_data), reference);
			const auto [input_grad_result, layer_grad_result] = nl.backpropagate(cost_gradient, aux_data);

			//Assert
			Assert::IsTrue(layer_grad_result.Biases_grad.size() == 0 &&
				layer_grad_result.Weights_grad.size() == 0, L"Unexpected output of the back-propagation procedure");
		}


		/// <summary>
		/// General method to run layer handle serialization test
		/// </summary>
		template <class L>
		static void RunSerializationTest(const L& layer)
		{
			//Act
			const auto layer_handle = LayerHandle::make<L>(layer);
			const auto msg = MsgPack::pack(layer_handle);
			const auto layer_handle_unpacked = MsgPack::unpack<LayerHandle>(msg);

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
			RunSerializationTest(ConstructStandardPLayer());
		}

		TEST_METHOD(CLayerSerializationTest)
		{
			RunSerializationTest(ConstructStandardCLayer());
		}

		TEST_METHOD(NLayerSerializationTest)
		{
			RunSerializationTest(NLayer(10, 23, ActivationFunctionId::SIGMOID));
		}
	};
}