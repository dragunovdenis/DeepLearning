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
#include <NeuralNet/UMLayer.h>
#include <NeuralNet/NLayer.h>
#include "StandardTestUtils.h"
#include <NeuralNet/MLayerHandle.h>
#include "MNetTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(UMLayerTest)
	{
		static constexpr auto _delta = std::is_same_v<Real, double> ? static_cast<Real>(1e-5) : static_cast<Real>(1e-2);
		static constexpr auto _tolerance = std::is_same_v<Real, double> ? static_cast<Real>(6e-10) : static_cast<Real>(3e-4);

		/// <summary>
		/// Instantiates a "standard" instance of the UMLayer for testing purposes.
		/// </summary>
		static UNMLayer<CpuDC> instantiate_layer()
		{
			constexpr auto depth = 5;
			const Index4d in_size = {{2, 2, 2,}, depth};
			const Index4d out_size = { {2, 1, 5}, depth };
			return UNMLayer<CpuDC>(in_size, out_size, ActivationFunctionId::SIGMOID);
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
			run_parameter_gradient_test(0);
		}

		TEST_METHOD(WeightGradientTest)
		{
			run_parameter_gradient_test(1);
		}

		TEST_METHOD(SerializationTest)
		{
			// Arrange
			const auto layer = instantiate_layer();

			// Act 
			const auto msg = MsgPack::pack(layer);
			const auto layer_unpacked = MsgPack::unpack<UNMLayer<CpuDC>>(msg);

			//Assert
			Assert::IsTrue(layer.equal(layer_unpacked), L"Original and restored layers are different");
		}

		TEST_METHOD(HandleSerializationTest)
		{
			// Arrange
			constexpr auto depth = 5;
			const Index4d in_size = { {2, 3, 7,}, depth };
			const Index4d out_size = { {5, 12, 4}, depth };
			const auto layer_handle = MLayerHandle<CpuDC>::make<UNMLayer>(in_size,
				out_size, ActivationFunctionId::TANH);

			// Act
			const auto msg = MsgPack::pack(layer_handle);
			const auto layer_handle_unpacked = MsgPack::unpack<MLayerHandle<CpuDC>>(msg);

			// Assert
			Assert::IsTrue(layer_handle == layer_handle_unpacked, L"Original and restored layers are different");
		}
	};
}
