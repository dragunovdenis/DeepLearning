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

#pragma once

#include "CppUnitTest.h"
#include <Math/ActivationFunction.h>
#include "StandardTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	/// <summary>
	/// Runs comparison test between the given function and an instance of
	/// `ActivationFunction` constructed from the given function identifier.
	/// </summary>
	template <class T>
	void RunOptimizedVsGeneralActivationFuncTest(const AFunction<T>& func, const ActivationFunctionId& func_id)
	{
		// Arrange
		const ActivationFunction<T> reference_func(func_id);
		const T input(Index3d{ 10, 20, 23 }, -1, 1);
		Assert::IsTrue(input.max_abs() > 0, L"Input vector is supposed to be nonzero");

		// Act
		const auto [value_0, derivative] = func.func_and_aux(input);
		const auto value_1 = func(input);

		// Assert
		Assert::IsTrue(value_0 == value_1, L"Value vectors produced by the same function must coincide");
		const auto [value_reference, derivative_reference] = reference_func.func_and_aux(input);
		const auto value_diff = (value_0 - value_reference).max_abs();
		StandardTestUtils::LogAndAssertLessOrEqualTo("Value difference", value_diff, 10 * std::numeric_limits<Real>::epsilon());
		const auto derivative_diff = (derivative - derivative_reference).max_abs();
		StandardTestUtils::LogAndAssertLessOrEqualTo("Derivative difference", derivative_diff, 10 * std::numeric_limits<Real>::epsilon());
	}
}
