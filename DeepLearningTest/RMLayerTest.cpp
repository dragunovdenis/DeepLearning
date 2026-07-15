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
#include <NeuralNet/MLayerHandle.h>
#include "MNetTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(RMLayerTest)
	{
		static constexpr auto _delta = std::is_same_v<Real, double> ? static_cast<Real>(1e-5) : static_cast<Real>(1e-2);
		static constexpr auto _tolerance = std::is_same_v<Real, double> ? static_cast<Real>(8e-10) : static_cast<Real>(3e-4);

		/// <summary>
		/// Instantiates a "standard" instance of the RMLayer for testing purposes.
		/// </summary>
		static RMLayer<CpuDC> instantiate_layer()
		{
			constexpr auto rec_depth = 5;
			constexpr auto in_size_plain = 10;
			const Index3d out_item_size(1, 1, 8);
			return RMLayer<CpuDC>(rec_depth, in_size_plain, out_item_size,
				FillRandomNormal, ActivationFunctionId::SIGMOID);
		}

		TEST_METHOD(InputGradientTest)
		{
			const auto layer = instantiate_layer();
			MNetTestUtils::run_standard_m_layer_input_gradient_test(layer, _delta, _tolerance);
		}

		/// <summary>
		/// General method to run gradient tests with respect to different parameter containers.
		/// </summary>
		static void run_parameter_gradient_test(const int param_container_id)
		{
			const auto layer = instantiate_layer();
			MNetTestUtils::run_standard_m_layer_parameter_gradient_test(layer, param_container_id, _delta, _tolerance);
		}

		TEST_METHOD(BiasesGradientTest)
		{
			run_parameter_gradient_test(RMLayer<CpuDC>::BIAS_GRAD_ID);
		}

		TEST_METHOD(InWeightGradientTest)
		{
			run_parameter_gradient_test(RMLayer<CpuDC>::IN_W_GRAD_ID);
		}

		TEST_METHOD(RecWeightGradientTest)
		{
			run_parameter_gradient_test(RMLayer<CpuDC>::REC_W_GRAD_ID);
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

		TEST_METHOD(HandleSerializationTest)
		{
			// Arrange
			const Index4d in_size{ {6, 9, 7}, 10 };
			const Index4d out_size{ {8, 11, 5}, 10 };
			const auto layer_handle = MLayerHandle<CpuDC>::make<RMLayer>(in_size, out_size, FillRandomNormal, ActivationFunctionId::SIGMOID);

			// Act
			const auto msg = MsgPack::pack(layer_handle);
			const auto layer_handle_unpacked = MsgPack::unpack<MLayerHandle<CpuDC>>(msg);

			// Assert

			Assert::IsTrue(layer_handle == layer_handle_unpacked, L"Original and restored layers are different");
		}
	};
}