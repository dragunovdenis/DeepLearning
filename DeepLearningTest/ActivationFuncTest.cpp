#include "CppUnitTest.h"
#include <Math/DenseVector.h>
#include <Math/ActivationFunction.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(ActivationfunctionTest)
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
			const auto result = Sigmoid()(vector);

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
			const auto [result, result_deriv] = Sigmoid().func_and_deriv(vector);

			//Assert
			Assert::AreEqual(vector.dim(), result.dim(), L"Unexpected dimension of the result vector");
			Assert::AreEqual(vector.dim(), result_deriv.dim(), L"Unexpected dimension of the result derivative vector");

			for (std::size_t item_id = 0; item_id < vector.dim(); item_id++)
			{
				const auto func_diff = std::abs(result(item_id) - sigmoid(vector(item_id)));
				Logger::WriteMessage((std::string("Function difference = ") + std::to_string(func_diff) + "\n").c_str());
				Assert::IsTrue(func_diff < std::numeric_limits<Real>::epsilon(),
					L"Unexpectedly high deviation from the function reference value.");

				const auto deriv_diff = std::abs(result_deriv(item_id) - sigmoid_deriv(vector(item_id)));
				Logger::WriteMessage((std::string("Derivative difference = ") + std::to_string(deriv_diff) + "\n").c_str());
				Assert::IsTrue(deriv_diff < std::numeric_limits<Real>::epsilon(),
					L"Unexpectedly high deviation from the derivative reference value.");
			}
		}
	};
}
