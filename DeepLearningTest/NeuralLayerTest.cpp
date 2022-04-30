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
#include <NeuralNet/NeuralLayer.h>
#include <Math/CostFunction.h>
#include <Utilities.h>
#include <type_traits>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NeuralLayerTest)
	{
		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to the input 
		/// </summary>
		void CheckDerivativeWithRespectToInputValuesCalculation(const ActivationFunctionId activation_func_id, const CostFunctionId cost_function_id)
		{
			const auto input_dim = 10;
			const auto output_dim = 23;
			const auto input_0 = Tensor(1, 1, input_dim, -1, 1);
			const auto weights_0 = Matrix(output_dim, input_dim, -1, 1);
			const auto biases_0 = Vector(output_dim, -1, 1);
			const auto reference = Tensor(1, 1, output_dim, -1, 1);;
			const auto cost_func = CostFunction(cost_function_id);
			const auto nl = NeuralLayer(weights_0, biases_0, activation_func_id);

			//Act
			auto aux_data = NeuralLayer::AuxLearningData();
			const auto [value, cost_gradient] = cost_func.func_and_deriv(nl.act(input_0, &aux_data), reference);
			const auto [input_grad_result, layer_grad_result] = nl.backpropagate(cost_gradient, aux_data);

			//Assert
			const auto nl_aux = NeuralLayer(weights_0, biases_0, activation_func_id);
			const auto delta = std::is_same_v<Real, double> ? Real(1e-5) : Real(1e-3);

			for (std::size_t in_item_id = 0; in_item_id < input_dim; in_item_id++)
			{
				//Calculate partial derivative with respect to the corresponding input item numerically
				auto input_minus_delta = input_0;
				input_minus_delta[in_item_id] -= delta;
				auto input_plus_delta = input_0;
				input_plus_delta[in_item_id] += delta;

				const auto result_minus = cost_func(nl_aux.act(input_minus_delta), reference);
				const auto result_plus = cost_func(nl_aux.act(input_plus_delta), reference);
				const auto deriv_numeric = (result_plus - result_minus) / (2 * delta);

				//Now do the same using the back-propagation approach
				const auto diff = std::abs(deriv_numeric - input_grad_result[in_item_id]);
				Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + '\n').c_str());
				Assert::IsTrue(diff <= (std::is_same_v<Real, double> ?  Real(1e-9) : Real(5e-3)), L"Unexpectedly high deviation!");
			}
		}

		/// <summary>
		/// General method to exercise back-propagation algorithm (for a single neural layer)
		/// in the part of calculating derivatives with respect to weights and biases
		/// </summary>
		void CheckDerivativeWithRespectToWeightsAndBiasesCalculation(const ActivationFunctionId activation_func_id, const CostFunctionId cost_function_id)
		{
			const auto input_dim = 10;
			const auto output_dim = 23;
			const auto input_0 = Tensor(1, 1, input_dim, -1, 1);
			const auto weights_0 = Matrix(output_dim, input_dim, -1, 1);
			const auto biases_0 = Vector(output_dim, -1, 1);
			const auto reference = Tensor(1, 1, output_dim, -1, 1);;
			const auto cost_func = CostFunction(cost_function_id);
			const auto nl = NeuralLayer(weights_0, biases_0, activation_func_id);

			//Act
			auto aux_data = NeuralLayer::AuxLearningData();
			const auto [value, cost_gradient] = cost_func.func_and_deriv(nl.act(input_0, &aux_data), reference);
			const auto [input_grad_result, layer_grad_result] = nl.backpropagate(cost_gradient, aux_data);
			const auto wight_grad = layer_grad_result.Weights_grad;
			const auto bias_grad  = layer_grad_result.Biases_grad;

			//Assert
			const auto nl_aux = NeuralLayer(weights_0, biases_0, activation_func_id);
			const auto delta = std::is_same_v<Real, double> ? Real(1e-5) : Real(1e-3);

			Assert::IsTrue(wight_grad.size() == 1, L"Unexpected size of the weight gradients vector");
			//Check derivatives with respect to weights
			Logger::WriteMessage("Weights:\n");
			for (std::size_t row_id = 0; row_id < output_dim; row_id++)
				for (std::size_t col_id = 0; col_id < input_dim; col_id++)
				{
					//Calculate partial derivative with respect to the corresponding weight
					auto weights_minus_delta = weights_0;
					weights_minus_delta(row_id, col_id) -= delta;
					auto weights_plus_delta = weights_0;
					weights_plus_delta(row_id, col_id) += delta;

					const auto result_minus = cost_func(NeuralLayer(weights_minus_delta, biases_0, activation_func_id).act(input_0), reference);
					const auto result_plus = cost_func(NeuralLayer(weights_plus_delta, biases_0, activation_func_id).act(input_0), reference);
					const auto deriv_numeric = (result_plus - result_minus) / (2 * delta);

					//Now do the same using the back-propagation approach
					const auto diff = std::abs(deriv_numeric - wight_grad[0](0, row_id, col_id));
					Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + '\n').c_str());
					Assert::IsTrue(diff <= (std::is_same_v<Real, double> ? Real(6e-10) : Real(3e-3)), L"Unexpectedly high deviation (weight derivatives)!");
				}

			//Check derivatives with respect to biases
			Logger::WriteMessage("Biases:\n");
			for (std::size_t row_id = 0; row_id < output_dim; row_id++)
			{
				//Calculate partial derivative with respect to the corresponding bias
				auto biases_minus_delta = biases_0;
				biases_minus_delta(row_id) -= delta;
				auto biases_plus_delta = biases_0;
				biases_plus_delta(row_id) += delta;

				const auto result_minus = cost_func(NeuralLayer(weights_0, biases_minus_delta, activation_func_id).act(input_0), reference);
				const auto result_plus = cost_func(NeuralLayer(weights_0, biases_plus_delta, activation_func_id).act(input_0), reference);
				const auto deriv_numeric = (result_plus - result_minus) / (2 * delta);

				//Now do the same using the back-propagation approach
				const auto diff = std::abs(deriv_numeric - bias_grad(0, 0, row_id));
				Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + '\n').c_str());
				Assert::IsTrue(diff <= (std::is_same_v<Real, double> ? Real(5e-10) : Real(3e-3)), L"Unexpectedly high deviation (bias derivatives)!");
			}
		}

		TEST_METHOD(DerivativeWithRespectToInputValuesCalculationSquaredErrorTest)
		{
			CheckDerivativeWithRespectToInputValuesCalculation(ActivationFunctionId::SIGMOID, CostFunctionId::SQUARED_ERROR);
		}

		TEST_METHOD(DerivativeWithRespectToInputValuesCalculationCrossEntropyTest)
		{
			CheckDerivativeWithRespectToInputValuesCalculation(ActivationFunctionId::SIGMOID, CostFunctionId::CROSS_ENTROPY);
		}

		TEST_METHOD(DerivativeWithRespectToWeightsAndbiasesCalculationSquaredErrorTest)
		{
			CheckDerivativeWithRespectToWeightsAndBiasesCalculation(ActivationFunctionId::SIGMOID, CostFunctionId::SQUARED_ERROR);
		}

		TEST_METHOD(DerivativeWithRespectToWeightsAndBiasesCalculationCrossEntropyTest)
		{
			CheckDerivativeWithRespectToWeightsAndBiasesCalculation(ActivationFunctionId::SIGMOID, CostFunctionId::CROSS_ENTROPY);
		}
	};
}