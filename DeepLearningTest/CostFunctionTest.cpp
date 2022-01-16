#include "CppUnitTest.h"
#include <Math/DenseVector.h>
#include <Math/CostFunction.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(CostFunctionTest)
	{
		/// <summary>
		/// Squared error reference function
		/// </summary>
		static Real SquaredErrorFunc(const Real x, const Real ref)
		{
			const auto diff = x - ref;
			return Real(0.5) * diff * diff;
		}

		/// <summary>
		/// Derivative of the squared error function
		/// </summary>
		static Real SquaredErrorDerivFunc(const Real x, const Real ref)
		{
			return x - ref;
		}

		/// <summary>
		/// Cross-entropy function
		/// </summary>
		static Real CrossEntropyFunc(const Real x, const Real ref)
		{
			return -(ref * log(x) + (Real(1) - ref) * log(1 - x));
		}

		/// <summary>
		/// Derivative of the cross-entropy function
		/// </summary>
		static Real CrossEntropyDerivFunc(const Real x, const Real ref)
		{
			return (x - ref) / (x * (Real(1) - x));
		}

		/// <summary>
		/// General method to test function only (without derivative) evaluation part of the cost function component
		/// </summary>
		template <class RF>
		static void CheckFunctionOnly(const CostFunctionId& func_id, const RF& reference_func)
		{
			//Arrange
			const auto dim = 10;
			const auto cost_func = CostFunction(func_id);
			const auto input = DenseVector(dim, 1e-5, 1-1e-5);
			const auto reference = DenseVector(dim, 1e-5, 1 - 1e-5);
			Assert::IsTrue(input != reference, L"Reference and input vectors should not be equal.");

			//Act
			const auto result = cost_func(input, reference);

			//Assert
			Assert::IsTrue(result.dim() == input.dim(), L"Input and result vectors should be of the same dimension.");

			for (std::size_t item_id = 0; item_id < dim; item_id++)
			{
				const auto diff = std::abs(result(item_id) - reference_func(input(item_id), reference(item_id)));
				Assert::IsTrue(diff <= 0, L"Unexpectedly high deviation from the reference value");
			}
		}

		/// <summary>
		/// General method to test function and derivative evaluation part of the cost function component
		/// </summary>
		template <class RF>
		static void CheckFunctionAndDerivative(const CostFunctionId& func_id, const RF& reference_func, const RF& reference_deriv)
		{
			//Arrange
			const auto dim = 10;
			const auto cost_func = CostFunction(func_id);
			const auto input = DenseVector(dim, 0.1, 0.9);
			const auto reference = DenseVector(dim, 0.1, 0.9);
			Assert::IsTrue(input != reference, L"Reference and input vectors should not be equal.");

			//Act
			const auto [result_func, result_deriv] = cost_func.func_and_deriv(input, reference);

			//Assert
			Assert::IsTrue(result_func.dim() == input.dim() && result_deriv.dim() == input.dim(), L"Input and result vectors should be of the same dimension.");

			for (std::size_t item_id = 0; item_id < dim; item_id++)
			{
				const auto diff_func = std::abs(result_func(item_id) - reference_func(input(item_id), reference(item_id)));
				Assert::IsTrue(diff_func <= 0, L"Unexpectedly high deviation from the reference function value.");

				const auto diff_deriv = std::abs(result_deriv(item_id) - reference_deriv(input(item_id), reference(item_id)));
				Assert::IsTrue(diff_deriv <= 10 * std::numeric_limits<Real>::epsilon(), L"Unexpectedly high deviation from the reference derivative value");
			}
		}

		TEST_METHOD(SquaredErrorFunctionTest)
		{
			CheckFunctionOnly(CostFunctionId::SQUARED_ERROR, SquaredErrorFunc);
		}

		TEST_METHOD(CrossEntropyFunctionTest)
		{
			CheckFunctionOnly(CostFunctionId::CROSS_ENTROPY, CrossEntropyFunc);
		}

		TEST_METHOD(SquaredErrorFunctionAndDerivativeTest)
		{
			CheckFunctionAndDerivative(CostFunctionId::SQUARED_ERROR, SquaredErrorFunc, SquaredErrorDerivFunc);
		}

		TEST_METHOD(CrossEntropyFunctionAndDerivativeTest)
		{
			CheckFunctionAndDerivative(CostFunctionId::CROSS_ENTROPY, CrossEntropyFunc, CrossEntropyDerivFunc);
		}
	};
}
