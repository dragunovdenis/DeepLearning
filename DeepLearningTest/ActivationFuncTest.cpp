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
#include <Math/DenseVector.h>
#include <Math/ActivationFunction.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(ActivationFunctionTest)
	{
	public:
		/// <summary>
		/// Sigmoid function
		/// </summary>
		static Real sigmoid(const Real& x)
		{
			return Real(1) / (Real(1) + std::exp(-x));
		}

		/// <summary>
		/// Derivative of the sigmoid function
		/// </summary>
		static Real sigmoid_deriv(const Real& x)
		{
			const auto exp_x = std::exp(-x);
			const auto denominator = (Real(1) + exp_x);
			return exp_x / (denominator * denominator);
		}

		TEST_METHOD(SigmoidFunctionTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector = DenseVector(dim, -1, 1);

			//Act
			const auto result = ActivationFuncion(ActivationFunctionId::SIGMOID)(vector);

			//Assert
			Assert::AreEqual(vector.dim(), result.dim(), L"Unexpected dimension of the result vector");

			for (std::size_t item_id = 0; item_id < vector.dim(); item_id++)
			{
				const auto diff = std::abs(result(item_id) - sigmoid(vector(item_id)));
				Logger::WriteMessage((std::string("Difference = ") + std::to_string(diff) + "\n").c_str());
				Assert::IsTrue(diff < std::numeric_limits<Real>::epsilon(),
					L"Unexpectedly high deviation from the reference value.");
			}
		}

		TEST_METHOD(SigmoidFunctionAndDerivativeTest)
		{
			//Arrange
			const auto dim = 10;
			const auto vector = DenseVector(dim, -1, 1);

			//Act
			const auto [result, result_deriv] = ActivationFuncion(ActivationFunctionId::SIGMOID).func_and_deriv(vector);

			//Assert
			Assert::AreEqual(vector.dim(), result.dim(), L"Unexpected dimension of the result vector");
			Assert::AreEqual(vector.dim(), result_deriv.dim(), L"Unexpected dimension of the result derivative vector");

			//Here we use the activation function to generate the reference values, because "()" operator of the activation function
			//is tested separately, and we rely on that
			const auto result_reference = ActivationFuncion(ActivationFunctionId::SIGMOID)(vector);
			Assert::IsTrue((result_reference - result).max_abs() <= 0, L"Unexpectedly high deviation from the function reference value.");

			for (std::size_t item_id = 0; item_id < vector.dim(); item_id++)
			{
				const auto deriv_diff = std::abs(result_deriv(item_id) - sigmoid_deriv(vector(item_id)));
				Logger::WriteMessage((std::string("Derivative difference = ") + std::to_string(deriv_diff) + "\n").c_str());
				Assert::IsTrue(deriv_diff < std::numeric_limits<Real>::epsilon(),
					L"Unexpectedly high deviation from the derivative reference value.");
			}
		}
	};
}
