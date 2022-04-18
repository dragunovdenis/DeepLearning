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

#pragma once

#include "CppUnitTest.h"
#include <MsgPackUtils.h>
#include "Utilities.h"
#include <functional>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

/// <summary>
/// Utility method to test some typical properties of classes
/// </summary>
namespace DeepLearningTest::StandardTestUtils
{
	/// <summary>
	/// Performs a "standard" test of message-pack serialization for the given class "T"
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void PackingTest(const std::function<T()>& factory)
	{
		//Arrange
		const auto instance = factory();
		Assert::IsTrue(!instance.empty(), L"Instances must be non-empty");

		//Act
		const auto msg = MsgPack::pack(instance);
		const auto instance_unpacked = MsgPack::unpack<T>(msg);

		//Assert
		Assert::IsTrue(instance == instance_unpacked, L"De-serialized instance is not equal to the original one.");
	}

	/// <summary>
	/// Performs a "standard" test of additive properties of zero element for the given class "T"
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	/// <param name="zero_instance">"Zero" instance of class "T"</param>
	template <class T>
	void SumWithZeroElementTest(const std::function<T()>& factory, const T& zero_instance)
	{
		//Arrange
		const auto instance = factory();
		Assert::IsTrue(instance.max_abs() > 0, L"The instance is zero!");
		Assert::IsTrue(zero_instance.max_abs() == 0, L"Zero instance is actually non-zero!");

		//Act
		const auto result_right = instance + zero_instance;
		const auto result_left = zero_instance + instance;

		//Assert
		Assert::IsTrue(instance == result_right &&
					   instance == result_left,
			L"Zero instance does not satisfy properties of zero element in an additive group!");
	}

	/// <summary>
	/// Performs a "standard" test of the commutativity property of addition operation 
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void AdditionCommutativityTest(const std::function<T()>& factory)
	{
		//Arrange
		const auto instance1 = factory();
		const auto instance2 = factory();

		Assert::IsTrue(instance1.max_abs() > 0 && instance2.max_abs() > 0, L"The two random instances are supposed to be non-zero!");
		Assert::IsTrue(instance1 != instance2, L"The two random instances are supposed to be different!");

		//Act
		const auto result1 = instance1 + instance2;
		const auto result2 = instance2 + instance1;

		//Assert
		Assert::IsTrue(result1 == result2, L"The addition operator is non-commutative!");
	}

	/// <summary>
	/// Performs a "standard" test of the subtraction operation: difference of two equal instances equals to the additive zero element
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void DifferenceOfEqualInstancesTest(const std::function<T()>& factory)
	{
		const auto instance1 = factory();
		const auto instance2 = instance1;
		Assert::IsTrue(instance1.max_abs() > 0, L"The instance is expected to be non-zero!");
		Assert::IsTrue(instance1 == instance2, L"The instances are not equal!");

		//Act
		const auto result = instance1 - instance2;

		//Assert
		Assert::IsTrue(result.max_abs() == 0, L"The result is non-zero");
	}

	/// <summary>
	/// Performs a "standard" test of the associativity of addition operator
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void AdditionAssociativityTest(const std::function<T()>& factory)
	{
		//Arrange
		const auto instance1 = factory();
		const auto instance2 = factory();
		const auto instance3 = factory();
		Assert::IsTrue(instance1.max_abs() > 0 &&
			instance2.max_abs() > 0 &&
			instance3.max_abs() > 0, L"The input instances are expected to be non-zero!");
		Assert::IsTrue(instance1 != instance2 && instance1 != instance3 && instance3 != instance2,
			L"The input instances are supposed to be different!");

		//Act
		const auto result1 = (instance1 + instance2) + instance3;
		const auto result2 = instance1 + (instance2 + instance3);

		//Assert
		Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
			L"Addition operator is non-associative!");
	}

	/// <summary>
	/// Performs a "standard" test of the scalar multiplication distributivity with respect to the addition operator
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void ScalarMultiplicationDistributivityTest(const std::function<T()>& factory)
	{
		//Arrange
		const auto instance1 = factory();
		const auto instance2 = factory();
		const auto scalar = Utils::get_random(-1, 1) + 2;
		Assert::IsTrue(instance1.max_abs() > 0 && instance2.max_abs() > 0,
			L"The input instances are expected to be non-zero!");
		Assert::IsTrue(scalar != 0, L"Scalar is expected to be non-zero!");

		//Act
		const auto result1 = (instance1 + instance2) * scalar;
		const auto result2 = instance1 * scalar + instance2 * scalar;

		//Assert
		Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
			L"Scalar multiplication operator is non-distributive with respect to the addition operator!");
	}

	/// <summary>
	/// Performs a "standard" test of the multiplication by "one"
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void ScalarMultiplicationByOneTest(const std::function<T()>& factory)
	{
		//Arrange
		const auto instance = factory();
		const auto scalar = Real(1);
		Assert::IsTrue(instance.max_abs() > 0, L"The input instance is expected to be non-zero!");

		//Act
		const auto result = instance * scalar;

		//Assert
		Assert::IsTrue(result == instance, L"Instances are not the same!");
	}

	/// <summary>
	/// Performs a "standard" test of a unary "minus" operator
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void UnaryMinusOperatorTest(const std::function<T()>& factory)
	{
		//Arrange
		const auto instance = factory();
		Assert::IsTrue(instance.max_abs() > 0, L"The input instance is expected to be non-zero!");

		//Act
		const auto minus_instance = -instance;
		const auto double_minus_instance = -minus_instance;

		//Assert
		Assert::IsTrue(instance == double_minus_instance, L"Minus operator should be inverse to itself!");
		Assert::IsTrue((instance + minus_instance).max_abs() == 0, L"Minus operator should result in an additionally inverse instance!");
	}
}