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

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NetLayersTest)
	{

		/// <summary>
		/// Generic method to construct a "standard" CLayer for tests (generic version).
		/// </summary>
		template <class D>
		static CLayer<D> CreateCLayer(const ActivationFunctionId activation)
		{
			const auto input_dim = Index3d(5, 13, 17);
			const auto filter_window = Index2d(3);
			constexpr auto filters_count = 7;
			const auto paddings = Index3d(0, 3, 6);
			const auto strides = Index3d(2);
			return CLayer<D>(input_dim, filter_window, filters_count, activation, paddings, strides);
		}

		/// <summary>
		/// Returns a "standard" CLayer instance to be used in testing (CUDA version)
		/// </summary>
		static CLayer<GpuDC> CreateCudaCLayer(const ActivationFunctionId activation = ActivationFunctionId::SIGMOID)
		{
			return CreateCLayer<GpuDC>(activation);
		}

		/// <summary>
		/// Returns a "standard" CLayer instance to be used in testing
		/// </summary>
		static CLayer<CpuDC> CreateCpuCLayer(const ActivationFunctionId activation = ActivationFunctionId::SIGMOID)
		{
			return CreateCLayer<CpuDC>(activation);
		}

		/// <summary>
		/// Returns a "standard" PLayer instance to be used in testing (generic version)
		/// </summary>
		template <class D>
		static PLayer<D> CreatePLayer(const PoolTypeId pool_oper_id)
		{
			const auto input_dim = Index3d(5, 10, 7);
			const auto filter_window = Index2d(3, 4);
			return PLayer<D>(input_dim, filter_window, pool_oper_id);
		}

		/// <summary>
		/// Returns a "standard" PLayer instance to be used in testing (CUDA version)
		/// </summary>
		static PLayer<GpuDC> CreateCudaPLayer(const PoolTypeId pool_oper_id)
		{
			return CreatePLayer<GpuDC>(pool_oper_id);
		}

		/// <summary>
		/// Returns a "standard" PLayer instance to be used in testing
		/// </summary>
		static PLayer<CpuDC> CreateCpuPLayer(const PoolTypeId pool_oper_id = PoolTypeId::MAX)
		{
			return CreatePLayer<CpuDC>(pool_oper_id);
		}

		/// <summary>
		/// Returns a "standard" NLayer instance to be used in testing (generic version)
		/// </summary>
		template <class D>
		static NLayer<D> CreateNLayer(const ActivationFunctionId activation_func_id)
		{
			const auto input_dim = 10;
			const auto output_dim = 23;
			return NLayer<D>(input_dim, output_dim, activation_func_id);
		}

		/// <summary>
		/// Returns a "standard" NLayer instance to be used in testing (CUDA version)
		/// </summary>
		static NLayer<GpuDC> CreateCudaNLayer(const ActivationFunctionId activation_func_id = ActivationFunctionId::SIGMOID)
		{
			return CreateNLayer<GpuDC>(activation_func_id);
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
		void RunGeneralDerivativeWithRespectToInputValuesTest(const ALayer<CpuDC>& nl, const CostFunctionId cost_function_id, const Real& tolerance,
			const Real& delta = std::is_same_v<Real, double> ? Real(1e-5) : Real(1e-3),
			const std::optional<Tensor>& input_op = std::nullopt)
		{
			const auto input = input_op.has_value() ? input_op.value() : Tensor(nl.in_size(), -1, 1);
			const auto reference = Tensor(nl.out_size(), -1, 1);;

			const auto cost_func = CostFunction<Tensor>(cost_function_id);
			auto aux_data = ALayer<CpuDC>::AuxLearningData();
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

			const auto reference = Tensor(out_size, -1, 1);
			const auto cost_func = CostFunction<Tensor>(cost_function_id);

			//Act
			auto aux_data = ALayer<CpuDC>::AuxLearningData();
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
							layer_minus.update(LayerGradient<CpuDC>{zero_biases, weights_minus_delta}, 1, 0);
							const auto result_minus = cost_func(layer_minus.act(input), reference);

							auto weights_plus_delta = zero_weights;
							weights_plus_delta[filter_id](layer_id, row_id, col_id) = delta;

							auto layer_plus = nl;//create a mutable copy of the initial layer
							layer_plus.update(LayerGradient<CpuDC>{zero_biases, weights_plus_delta}, 1, 0);
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
						layer_minus.update(LayerGradient<CpuDC>{biases_minus_delta, zero_weights}, 1, 0);
						const auto result_minus = cost_func(layer_minus.act(input), reference);

						auto biases_plus_delta = zero_biases;
						biases_plus_delta(layer_id, row_id, col_id) = delta;

						auto layer_plus = nl;//create a mutable copy of the initial layer
						layer_plus.update(LayerGradient<CpuDC>{biases_plus_delta, zero_weights}, 1, 0);
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
		/// General method to test gradient with scaling factor calculation
		/// </summary>
		template <template <typename> class L, class D>
		void RunGeneralGradientWithScalingTest(const L<D>& nl)
		{
			const typename D::tensor_t input(nl.in_size(), static_cast<Real>(-1), static_cast<Real>(1));
			typename D::tensor_t input_grad_result(nl.in_size(), /*fill zeros*/ true);
			auto aux_data = ALayer<D>::AuxLearningData();
			LayerGradient<D> gradient_container;
			nl.allocate(gradient_container, /*fill zeros*/ false);

			gradient_container.Biases_grad.standard_random_fill();
			for (auto& filter_gradient : gradient_container.Weights_grad)
				filter_gradient.standard_random_fill();

			const auto gradient_container_input = gradient_container;
			const auto gradient_scale_factor = Utils::get_random(-1, 1);

			//Act
			const auto output = nl.act(input, &aux_data);
			nl.backpropagate(output, aux_data, input_grad_result, gradient_container,
				/*evaluate_input_gradient*/ true, gradient_scale_factor);

			// Assert
			const auto [reference_input_grad_result, reference_layer_grad_result] = nl.backpropagate(output, aux_data);
			const auto diff = (gradient_container_input * gradient_scale_factor +
				reference_layer_grad_result - gradient_container).max_abs();
			Logger::WriteMessage((std::string("Gradient discrepancy = ") + Utils::to_string(diff) + '\n').c_str());
			Assert::IsTrue(diff < 10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from reference");
			const auto input_grad_diff = (reference_input_grad_result - input_grad_result).max_abs();
			Logger::WriteMessage((std::string("Input gradient discrepancy = ") + Utils::to_string(input_grad_diff) + '\n').c_str());
			Assert::IsTrue(input_grad_diff < 10 * std::numeric_limits<Real>::epsilon(),
				L"Input gradient must not be affected by scaling factor");
		}

		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to the input 
		/// </summary>
		void CheckDerivativeWithRespectToInputValuesCalculation(const ActivationFunctionId activation_func_id, const CostFunctionId cost_function_id)
		{
			const auto nl = CreateCpuNLayer(activation_func_id);
			RunGeneralDerivativeWithRespectToInputValuesTest(nl, cost_function_id, (std::is_same_v<Real, double> ? Real(2e-9) : Real(5e-3)));
		}

		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to weights and biases
		/// </summary>
		void CheckDerivativeWithRespectToWeightsAndBiasesCalculation(const ActivationFunctionId activation_func_id, const CostFunctionId cost_function_id)
		{
			const auto nl = CreateCpuNLayer(activation_func_id);
			RunGeneralDerivativeWithRespectToWeightsAndBiasesTest(nl, cost_function_id,
				(std::is_same_v<Real, double> ? Real(8e-10) : Real(3e-3)),
				(std::is_same_v<Real, double> ? Real(7e-10) : Real(3e-3)));
		}

		/// <summary>
		/// Returns L-infinity norm of the difference between the given 4D tensors
		/// </summary>
		Real max_abs(const std::vector<Tensor>& v1, const std::vector<CudaTensor>& v2)
		{
			if (v1.size() != v2.size())
				throw std::exception("The input vectors must be of the same size");

			auto result = Real(0);

			for (auto tensor_id = 0ull; tensor_id < v1.size(); tensor_id++)
				result = std::max(result, (v1[tensor_id] - v2[tensor_id].to_host()).max_abs());

			return result;
		}

		/// <summary>
		/// General method to compare CUDA accelerated NLayer implementation with the "usual" one
		/// </summary>
		template <template<class> class L>
		void LayerCudaSupportTest(const std::function<L<GpuDC>()> layer_factory)
		{
			//Arrange
			const auto nl = layer_factory();
			const auto input = CudaTensor(nl.in_size(), -1, 1);
			const auto out_gradient = CudaTensor(nl.out_size(), -1, 1);
			typename ALayer<GpuDC>::AuxLearningData aux_learning_data;
			Assert::IsTrue(input.max_abs() > 0 && out_gradient.max_abs() > 0, L"Input tensors are supposed to be nonzero");

			//Act
			const auto output = nl.act(input, &aux_learning_data);
			const auto [in_gradient, layer_gradient] = nl.backpropagate(out_gradient, aux_learning_data, true);

			//Assert
			Assert::IsTrue(output.max_abs() > 0 && in_gradient.max_abs() > 0, L"Output tensors are supposed to be nonzero");//A sanity check
			const auto nl_host = nl.to_host().to_device().to_host();//involve "to_device" into the testing as well
			const auto input_host = input.to_host();
			const auto out_gradient_host = out_gradient.to_host();
			typename ALayer<CpuDC>::AuxLearningData aux_learning_data_host;

			const auto output_host = nl_host.act(input_host, &aux_learning_data_host);
			const auto [in_gradient_host, layer_gradient_host] = nl_host.backpropagate(out_gradient_host, aux_learning_data_host, true);

			const auto output_diff = (output_host - output.to_host()).max_abs();
			StandardTestUtils::LogAndAssertLessOrEqualTo("output_diff", output_diff, 10* std::numeric_limits<Real>::epsilon());

			const auto in_gradient_diff = (in_gradient_host - in_gradient.to_host()).max_abs();
			StandardTestUtils::LogAndAssertLessOrEqualTo("in_gradient_diff", in_gradient_diff, 10 * std::numeric_limits<Real>::epsilon());

			if (!layer_gradient_host.Biases_grad.empty() || !layer_gradient.Biases_grad.empty())
			{
				const auto biases_gradient_diff = (layer_gradient_host.Biases_grad - layer_gradient.Biases_grad.to_host()).max_abs();
				StandardTestUtils::LogAndAssertLessOrEqualTo("biases_gradient_diff", biases_gradient_diff, 10 * std::numeric_limits<Real>::epsilon());
			}

			const auto weights_gradient_diff = max_abs(layer_gradient_host.Weights_grad, layer_gradient.Weights_grad);
			StandardTestUtils::LogAndAssertLessOrEqualTo("weights_gradient_diff", weights_gradient_diff, 10 * std::numeric_limits<Real>::epsilon());

			if (!aux_learning_data_host.Derivatives.empty() || !aux_learning_data.Derivatives.empty())
			{
				const auto deriv_diff = (aux_learning_data_host.Derivatives - aux_learning_data.Derivatives.to_host()).max_abs();
				StandardTestUtils::LogAndAssertLessOrEqualTo("deriv_diff", deriv_diff, 50 * std::numeric_limits<Real>::epsilon());
			}

			const auto indices_are_equal = aux_learning_data_host.IndexMapping == aux_learning_data.IndexMapping.to_stdvector();
			StandardTestUtils::Log("indices_are_equal", indices_are_equal);
			Assert::IsTrue(indices_are_equal, L"Arrays of indices are not equal");

			const auto sum_of_weight_squares_diff = std::abs(nl.squared_weights_sum() - nl_host.squared_weights_sum());
			StandardTestUtils::LogAndAssertLessOrEqualTo("sum_of_weight_squares_diff", sum_of_weight_squares_diff,
				1000 * std::numeric_limits<Real>::epsilon());
		}

		TEST_METHOD(NLayerSigmoidCudaSupportTest)
		{
			LayerCudaSupportTest<NLayer>([]() { return CreateCudaNLayer(ActivationFunctionId::SIGMOID); });
		}

		TEST_METHOD(NLayerSoftMaxCudaSupportTest)
		{
			LayerCudaSupportTest<NLayer>([]() { return CreateCudaNLayer(ActivationFunctionId::SOFTMAX); });
		}

		TEST_METHOD(CLayerSigmoidCudaSupportTest)
		{
			LayerCudaSupportTest<CLayer>([]() { return CreateCudaCLayer(ActivationFunctionId::SIGMOID); });
		}

		TEST_METHOD(CLayerSoftMaxCudaSupportTest)
		{
			LayerCudaSupportTest<CLayer>([]() { return CreateCudaCLayer(ActivationFunctionId::SOFTMAX); });
		}

		TEST_METHOD(PLayerMaxCudaSupportTest)
		{
			LayerCudaSupportTest<PLayer>([]() { return CreateCudaPLayer(PoolTypeId::MAX); });
		}

		TEST_METHOD(PLayerAverageCudaSupportTest)
		{
			LayerCudaSupportTest<PLayer>([]() { return CreateCudaPLayer(PoolTypeId::AVERAGE); });
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

		TEST_METHOD(NLayerGradientWithScalingCudaTest)
		{
			RunGeneralGradientWithScalingTest(CreateCudaNLayer());
		}

		TEST_METHOD(CLayerGradientWithScalingTest)
		{
			RunGeneralGradientWithScalingTest(CreateCpuCLayer());
		}

		TEST_METHOD(CLayerGradientWithScalingCudaTest)
		{
			RunGeneralGradientWithScalingTest(CreateCudaCLayer());
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
				(std::is_same_v<Real, double> ? static_cast<Real>(1e-10) : static_cast<Real>(5e-4)),
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
			auto aux_data = NLayer<CpuDC>::AuxLearningData();
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