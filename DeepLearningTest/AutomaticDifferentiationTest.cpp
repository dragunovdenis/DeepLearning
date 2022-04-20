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
#include <Math/Dual.h>
#include <functional>
#include <Utilities.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(AutomaticDifferentiationTest)
	{
		/// <summary>
		/// A general method to perform test of automatic differentiation of functions of one variable
		/// </summary>
		template <class Func, class R>
		static void RunOneVariableDifferentiationTest(const Func& one_var_function, const R& arg, const R& diff_rolerance)
		{
			//Arrange
			const R arg_step = R(1e-5);
			const auto deriv_numeric = (one_var_function(arg + arg_step) - one_var_function(arg - arg_step)) / (R(2) * arg_step);

			//Act
			const auto arg_dual = dual<R>(arg, { 1 });
			const auto result_dual = one_var_function(arg_dual);
			const auto deriv_automatic = result_dual.Dual()[0];

			//Assert
			const auto diff = std::abs(deriv_numeric - deriv_automatic);
			Logger::WriteMessage((std::string("Difference = ") + Utils::to_string(diff) + "\n").c_str());
			Assert::IsTrue(diff < diff_rolerance, L"too high deviation from reference");
		}

		/// <summary>
		/// A general method to perform test of automatic differentiation of functions of two variables
		/// </summary>
		template <class Func, class R>
		static void RunTwoVariablesDifferentiationTest(const Func& two_var_function, const R& arg1, const R& arg2, const R& diff_rolerance)
		{
			//Arrange
			const R arg_step = R(1e-5);
			const auto deriv1_numeric = (two_var_function(arg1 + arg_step, arg2) -
				two_var_function(arg1 - arg_step, arg2)) / (R(2) * arg_step);
			const auto deriv2_numeric = (two_var_function(arg1, arg2 + arg_step) -
				two_var_function(arg1, arg2 - arg_step)) / (R(2) * arg_step);

			//Act
			const auto arg1_dual = dual<R, 2>(arg1, { 1, 0 });
			const auto arg2_dual = dual<R, 2>(arg2, { 0, 1 });
			const auto result_dual = two_var_function(arg1_dual, arg2_dual);
			const auto deriv1_automatic = result_dual.Dual()[0];
			const auto deriv2_automatic = result_dual.Dual()[1];

			//Assert
			const auto diff1 = std::abs(deriv1_numeric - deriv1_automatic);
			const auto diff2 = std::abs(deriv2_numeric - deriv2_automatic);
			Logger::WriteMessage((std::string("Difference 1 = ") + Utils::to_string(diff1) + "\n").c_str());
			Logger::WriteMessage((std::string("Difference 2 = ") + Utils::to_string(diff2) + "\n").c_str());
			Assert::IsTrue(diff1 < diff_rolerance, L"too high deviation from reference");
			Assert::IsTrue(diff2 < diff_rolerance, L"too high deviation from reference");
		}

		TEST_METHOD(SingleVarPolynomialFunctionsTest)
		{
			const auto arg = Utils::get_random(-1, 1);
			RunOneVariableDifferentiationTest(
				[](const auto& x) { 
					return x * x * x * x * Real(5) - 3 * x * x * x + x / Real(7.8) * x + Real(1.5) * x - 2; 
				}, arg, Real(3e-9));
		}

		TEST_METHOD(SingleVarRationalFunctionsTest)
		{
			const auto arg = Utils::get_random(-1, 1);
			RunOneVariableDifferentiationTest(
				[](const auto& x) {
					return (x * x * x * x * Real(5) - 3 * x * x * x + x / Real(7.8) * x + Real(1.5) * x - 2)/(x*x + 4*x + 4);
				}, arg, Real(2e-8));
		}

		TEST_METHOD(SingleVarTrigonometricFunctionsTest)
		{
			const auto arg = Utils::get_random(-1, 1);
			RunOneVariableDifferentiationTest(
				[](const auto& x) {
					return sin(2*x + 3)*cos(3-3*x*x) * x - cos(3 - 3 * x * x * x)/(x * x + 1) + 2;
				}, arg, Real(9e-9));
		}

		TEST_METHOD(SingleVarLogarithmicFunctionsTest)
		{
			const auto arg = Utils::get_random(-1, 1);
			RunOneVariableDifferentiationTest(
				[](const auto& x) {
					return log(x * x + 1) + 2;
				}, arg, Real(1e-10));
		}

		TEST_METHOD(SingleVarExponentFunctionsTest)
		{
			const auto arg = Utils::get_random(-1, 1);
			RunOneVariableDifferentiationTest(
				[](const auto& x) {
					return exp(x * x - 1) + 2;
				}, arg, Real(4e-10));
		}

		TEST_METHOD(SingleVarSquareRootFunctionsTest)
		{
			const auto arg = Utils::get_random(-1, 1);
			RunOneVariableDifferentiationTest(
				[](const auto& x) {
					return sqrt(x * x + 4 * x + 4) + 2;
				}, arg, Real(1e-10));
		}

		TEST_METHOD(SingleVarHyperbolicFunctionsTest)
		{
			const auto arg = Utils::get_random(-1, 1);
			RunOneVariableDifferentiationTest(
				[](const auto& x) {
					return sinh(x * x + 2*x + 1) + cosh(-x*x) + tanh(1 + x) + 2;
				}, arg, Real(4e-8));
		}

		TEST_METHOD(TwoVarGeneralFunctionsTest)
		{
			const auto arg1 = Utils::get_random(-1, 1);
			const auto arg2 = Utils::get_random(-1, 1);
			RunTwoVariablesDifferentiationTest(
				[](const auto& x, const auto& y) {
					return x * x * x * x * Real(5) + 2 * y * y - 3 * x * x * y - 2 + sin(x + 2 * y) +
						cos(x * y) + exp(y / 2 + x / 2) + log(1 + x * x + y * y) + sqrt(cosh((x + y) / 10)) + sinh(x + y) / (1 + tanh(x * y)); 
				}, arg1, arg2, Real(2.1e-9));
		}
	};
}
