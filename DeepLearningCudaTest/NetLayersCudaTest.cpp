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

#include "CppUnitTest.h"
#include <NeuralNet/NLayer.h>
#include <NeuralNet/CLayer.h>
#include <NeuralNet/PLayer.h>
#include "StandardTestUtils.h"
#include "NetLayersTestUtils.h"
#include <NeuralNet/DataContextCuda.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NetLayersCudaTest)
	{

		/// <summary>
		/// Returns a "standard" CLayer instance to be used in testing (CUDA version)
		/// </summary>
		static CLayer<GpuDC> CreateCudaCLayer(const ActivationFunctionId activation = ActivationFunctionId::SIGMOID)
		{
			return CreateCLayer<GpuDC>(activation);
		}

		/// <summary>
		/// Returns a "standard" PLayer instance to be used in testing (CUDA version)
		/// </summary>
		static PLayer<GpuDC> CreateCudaPLayer(const PoolTypeId pool_oper_id)
		{
			return CreatePLayer<GpuDC>(pool_oper_id);
		}

		/// <summary>
		/// Returns a "standard" NLayer instance to be used in testing (CUDA version)
		/// </summary>
		static NLayer<GpuDC> CreateCudaNLayer(const ActivationFunctionId activation_func_id = ActivationFunctionId::SIGMOID)
		{
			return CreateNLayer<GpuDC>(activation_func_id);
		}

		/// <summary>
		/// Returns L-infinity norm of the difference between the given 4D tensors
		/// </summary>
		static Real max_abs(const std::vector<Tensor>& v1, const std::vector<CudaTensor>& v2, const std::size_t skip_count)
		{
			if (v1.size() != v2.size())
				throw std::exception("The input vectors must be of the same size");

			auto result = Real(0);

			for (auto tensor_id = skip_count; tensor_id < v1.size(); tensor_id++)
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
			LayerData<GpuDC> layer_data(input);
			Assert::IsTrue(input.max_abs() > 0 && out_gradient.max_abs() > 0, L"Input tensors are supposed to be nonzero");

			//Act
			const auto output = nl.act(layer_data.Input, &layer_data.Trace);
			const auto [in_gradient, layer_gradient] = nl.backpropagate(out_gradient, layer_data, true);

			//Assert
			Assert::IsTrue(output.max_abs() > 0 && in_gradient.max_abs() > 0, L"Output tensors are supposed to be nonzero");//A sanity check
			const auto nl_host = nl.template convert<CpuDC>().template convert<GpuDC>().template convert<CpuDC>();//involve "to_device" into the testing as well
			const auto input_host = input.to_host();
			const auto out_gradient_host = out_gradient.to_host();
			LayerData<CpuDC> layer_data_host(input_host);

			const auto output_host = nl_host.act(layer_data_host.Input, &layer_data_host.Trace);
			const auto [in_gradient_host, layer_gradient_host] = nl_host.backpropagate(out_gradient_host, layer_data_host, true);

			const auto output_diff = (output_host - output.to_host()).max_abs();
			StandardTestUtils::LogAndAssertLessOrEqualTo("output_diff", output_diff, 10* std::numeric_limits<Real>::epsilon());

			const auto in_gradient_diff = (in_gradient_host - in_gradient.to_host()).max_abs();
			StandardTestUtils::LogAndAssertLessOrEqualTo("in_gradient_diff", in_gradient_diff, 10 * std::numeric_limits<Real>::epsilon());

			if (!layer_gradient_host.empty() || !layer_gradient.empty())
			{
				const auto biases_gradient_diff = (layer_gradient_host.data[0] - layer_gradient.data[0].to_host()).max_abs();
				StandardTestUtils::LogAndAssertLessOrEqualTo("biases_gradient_diff", biases_gradient_diff, 10 * std::numeric_limits<Real>::epsilon());
			}

			const auto weights_gradient_diff = max_abs(layer_gradient_host.data, layer_gradient.data, /*skip count*/ 1);
			StandardTestUtils::LogAndAssertLessOrEqualTo("weights_gradient_diff", weights_gradient_diff, 10 * std::numeric_limits<Real>::epsilon());

			const auto& layer_trace = layer_data.Trace;
			const auto& layer_trace_host = layer_data_host.Trace;

			if (!layer_trace_host.Derivatives.empty() || !layer_trace.Derivatives.empty())
			{
				const auto deriv_diff = (layer_trace_host.Derivatives - layer_trace.Derivatives.to_host()).max_abs();
				StandardTestUtils::LogAndAssertLessOrEqualTo("deriv_diff", deriv_diff, 50 * std::numeric_limits<Real>::epsilon());
			}

			const auto indices_are_equal = layer_trace_host.IndexMapping == layer_trace.IndexMapping.to_stdvector();
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

		TEST_METHOD(NLayerGradientWithScalingCudaTest)
		{
			RunGeneralGradientWithScalingTest(CreateCudaNLayer());
		}

		TEST_METHOD(CLayerGradientWithScalingCudaTest)
		{
			RunGeneralGradientWithScalingTest(CreateCudaCLayer());
		}
	};
}
