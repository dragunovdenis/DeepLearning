//Copyright (c) 2023 Denys Dragunov, dragunovdenis@gmail.com
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
#include "StandardTestUtils.h"
#include "NeuralNet/LayerGradient.h"
#include "NeuralNet/DataContext.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	/// <summary>
	///	Factory method to generate random (or zero, depending on the input parameter) instance of "layer gradient" structure
	/// </summary>
	LayerGradient<CpuDC> layer_gradient_factory(const bool assign_zero = false)
	{
		auto result = LayerGradient<CpuDC>{ CpuDC::tensor_t(10, 15, 23, true),
			{ CpuDC::tensor_t(1, 2, 3, true),
				CpuDC::tensor_t(4, 5, 6, true),
				CpuDC::tensor_t(7, 8, 9, true)
		} };

		if (assign_zero)
			return result;

		result.Biases_grad.standard_random_fill();
		for (auto& item : result.Weights_grad)
			item.standard_random_fill();

		return result;
	}

	TEST_CLASS(LayerGradientTest)
	{
		TEST_METHOD(PackingTest)
		{
			StandardTestUtils::PackingTest<LayerGradient<CpuDC>>([]() { return layer_gradient_factory(); });
		}

		TEST_METHOD(SumWithZeroElementTest)
		{
			StandardTestUtils::SumWithZeroElementTest<LayerGradient<CpuDC>>(
				[]() { return layer_gradient_factory(); }, layer_gradient_factory(true));
		}

		TEST_METHOD(AdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<LayerGradient<CpuDC>>([]() { return layer_gradient_factory(); });
		}

		TEST_METHOD(DifferenceOfEqualGradientsIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<LayerGradient<CpuDC>>([]() { return layer_gradient_factory(); });
		}

		TEST_METHOD(AdditionAssociativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<LayerGradient<CpuDC>>([]() { return layer_gradient_factory(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWuthRespectToAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<LayerGradient<CpuDC>>([]() { return layer_gradient_factory(); });
		}

		TEST_METHOD(MultiplecationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<LayerGradient<CpuDC>>([]() { return layer_gradient_factory(); });
		}

		TEST_METHOD(AddScaledTest)
		{
			//Arrange
			const auto gradient0 = layer_gradient_factory();
			const auto gradient1 = layer_gradient_factory();
			const auto scalar = Utils::get_random(1.0, 5.0);

			//Sanity checks
			Assert::IsTrue(gradient0.max_abs() > 0 && gradient1.max_abs() > 0, L"Gradients are supposed to be nonzero");
			Assert::IsTrue(gradient0 != gradient1, L"Gradients are supposed to be different");

			//Act
			auto gradient = gradient0;
			gradient.add_scaled(gradient1, scalar);

			//Assert
			const auto reference = gradient0 + gradient1 * scalar;

			const auto diff = (gradient - reference).max_abs();
			StandardTestUtils::LogAndAssertLessOrEqualTo("Difference", diff, static_cast<Real>(0));
		}

		TEST_METHOD(ScaleAndAddTest)
		{
			//Arrange
			const auto gradient0 = layer_gradient_factory();
			const auto gradient1 = layer_gradient_factory();
			const auto scalar = Utils::get_random(1.0, 5.0);

			//Sanity checks
			Assert::IsTrue(gradient0.max_abs() > 0 && gradient1.max_abs() > 0, L"Gradients are supposed to be nonzero");
			Assert::IsTrue(gradient0 != gradient1, L"Gradients are supposed to be different");

			//Act
			auto gradient = gradient0;
			gradient.scale_and_add(scalar, gradient1);

			//Assert
			const auto reference = gradient0 * scalar + gradient1;

			const auto diff = (gradient - reference).max_abs();
			StandardTestUtils::LogAndAssertLessOrEqualTo("Difference", diff, static_cast<Real>(0));
		}
	};
}
