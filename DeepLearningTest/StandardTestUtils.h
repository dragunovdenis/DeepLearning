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
#include <chrono>
#include <atlbase.h>
#include <atlconv.h>

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

	/// <summary>
	/// Performs a "standard" test of the copy constructor of class "T"
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void CopyConstructorTest(const std::function<T()>& factory)
	{
		//Arrange
		const auto dim = 10;
		const auto instance = factory();

		//Act
		const T instance_copy(instance);

		//Assert
		Assert::IsTrue(instance == instance_copy, L"Instances are not the same");
		Assert::IsTrue(instance.begin() != instance_copy.begin(), L"Instances share the same memory");
	}

	/// <summary>
	/// Performs "standard" test of copy operator of class "T"
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	/// <param name="factory_diff_size">A factory method returning random instances of class "T"
	/// occupying different amount of memory than those instantiated by "factory"</param>
	template <class T>
	void AssignmentOperatorTest(const std::function<T()>& factory, const std::function<T()>& factory_diff_size)
	{
		//Arrange
		auto inst_to_assign = factory();
		const auto ptr_before_assignment = inst_to_assign.begin();

		auto inst_to_assign1 = factory();
		const auto ptr_before_assignment1 = inst_to_assign1.begin();

		const auto inst_to_copy = factory();
		const auto inst_to_copy1 = factory_diff_size();

		Assert::IsTrue(inst_to_assign != inst_to_copy && inst_to_assign != inst_to_copy1,
			L"Instances are supposed to be different");

		Assert::IsTrue(inst_to_assign1.size() != inst_to_copy1.size(),
			L"Instances are supposed to be of different sizes");

		//Act
		inst_to_assign = inst_to_copy;//Assign instance of the same size
		inst_to_assign1 = inst_to_copy1;//Assign instance of different size

		//Assert
		Assert::IsTrue(inst_to_assign == inst_to_copy, L"Copying failed (same size)");
		Assert::IsTrue(ptr_before_assignment == inst_to_assign.begin(), L"Memory was re-allocated when copying instances of the same size");

		Assert::IsTrue(inst_to_assign1 == inst_to_copy1, L"Copying failed (different sizes)");
		Assert::IsTrue(inst_to_assign1.begin() != inst_to_copy1.begin(), L"Instances share the same memory");
	}

	/// <summary>
	/// Performs a "standard" move constructor test of class "T"
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void MoveConstructorTest(const std::function<T()>& factory)
	{
		//Arrange
		auto instance_to_move = factory();
		const auto begin_pointer_before_move = instance_to_move.begin();
		const auto end_pointer_before_move = instance_to_move.end();

		//Act
		const T instance(std::move(instance_to_move));

		//Assert
		Assert::IsTrue(begin_pointer_before_move == instance.begin()
			&& end_pointer_before_move == instance.end(), L"Move operator does not work as expected");

		Assert::IsTrue(instance_to_move.begin() == nullptr && instance_to_move.size() == 0,
			L"Unexpected state for a vector after being moved");
	}

	template <class T>
	T mixed_arithmetic_function(const T& v1, const T& v2, const T& v3)
	{
		return (0.5 * v1 + v1 * 3.4 - 5.1 * v3) * 0.75;
	}

	/// <summary>
	/// General test for basic arithmetic operations of class T. It is assumed that
	/// any instance of class T can be converted to its "host" counterpart by calling "to_host()" method.
	/// The test is conducted with respect to the results of the "host" implementation
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void CudaMixedArithmeticTest(const std::function<T()>& factory)
	{
		//Arrange
		const auto inst1 = factory();
		const auto inst2 = factory();
		const auto inst3 = factory();

		Assert::IsTrue(inst1 != inst2 &&
			inst1 != inst3 && inst2 != inst3, L"Vectors are supposed to be different");

		//Act
		const auto result = mixed_arithmetic_function(inst1, inst2, inst3);

		//Assert
		const auto host_result = mixed_arithmetic_function(
			inst1.to_host(), inst2.to_host(), inst3.to_host());

		Assert::IsTrue((result.to_host() - host_result).max_abs() <
			10 * std::numeric_limits<Real>::epsilon(),
			L"Too high deviation from reference");
	}

	/// <summary>
	/// "MaxAbs" method of class T. It is assumed that
	/// any instance of class T can be converted to its "host" counterpart by calling "to_host()" method.
	/// The test is conducted with respect to the results of the "host" implementation
	/// </summary>
	/// <param name="factory">A factory method returning random instances of class "T"</param>
	template <class T>
	void CudaMaxAbsTest(const std::function<T()>& factory)
	{
		//Arrange
		const auto inst = factory();

		//Act
		const auto max_abs = inst.max_abs();

		//Assert
		Assert::IsTrue(max_abs == inst.to_host().max_abs(), L"Actual and expected values are not equal");
		Assert::IsTrue(max_abs > 0, L"Vector is supposed to be nonzero");
	}

	/// <summary>
	/// Executes the given action for the given number of iterations and
	/// returns average execution time in milliseconds
	/// </summary>
	/// <param name="action">Action method to measure execution time</param>
	/// <param name="number_of_iterations">Number of iterations the action
	/// should be executed in order to calculate the average execution time</param>
	template <class A>
	double ReportExecutionTime(const A& action, const std::size_t& number_of_iterations)
	{
		const auto start = std::chrono::steady_clock::now();

		for (auto iter_id = 0ull; iter_id < number_of_iterations; iter_id++)
			action();

		const auto end = std::chrono::steady_clock::now();

		return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1.0e-3 /number_of_iterations;
	}

	/// <summary>
	/// Logs given real value together with the given message in a single line
	/// </summary>
	inline void LogReal(const std::string& message, const Real& value)
	{
		Logger::WriteMessage((message + " = " + Utils::to_string(value) + "\n").c_str());
	}

	/// <summary>
	/// Logs given real value together with the given message in a single line
	/// </summary>
	inline void LogRealAndAssertLessOrEqualTo(const std::string& message, const Real& value, const Real& upper_threshold)
	{
		LogReal(message, value);
		Assert::IsTrue(value <= upper_threshold, CA2W((message + ": " + std::string("Upper threshold exceeded.")).c_str()));
	}

}