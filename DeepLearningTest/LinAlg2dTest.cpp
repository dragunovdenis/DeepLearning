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
#include <Math/LinAlg2d.h>
#include <set>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <Utilities.h>
#include <numbers>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(LinAlg2dTest)
	{
		/// <summary>
		/// Calculates mean and standard deviation of the values from the given set
		/// </summary>
		std::tuple<Real, Real> calc_mean_and_std(const std::set<Real>& set)
		{
			const auto mean = std::accumulate(set.begin(), set.end(), Real(0.0)) / set.size();
			double stdev = std::sqrt(std::inner_product(set.begin(), set.end(), set.begin(), Real(0.0)) / set.size() - mean * mean);

			return std::make_tuple(mean, Real(stdev));
		}

		/// <summary>
		/// Performs basic validation of the given collection of uniformly distributed on [-1, 1] random values
		/// </summary>
		void check_random_values(const std::set<Real>& set)
		{
			//We expect that the coordinates of random vectors take values in (-1, 1)
			Assert::IsTrue(std::all_of(set.begin(), set.end(), [](const auto x) { return x >= Real(-1) && x <= Real(1); }), L"Not all random values belong to [-1, 1].");

			const auto [mean, stdev] = calc_mean_and_std(set);

			Logger::WriteMessage((std::string("Mean = ") + Utils::to_string(mean) + "; Standard deviation = " + Utils::to_string(stdev) + "\n").c_str());
			//We expect that the random values follows uniform distribution on [-1, 1], thus their mean should tend to "0" and standard deviation should be close to sqrt(1/3);
			const auto mean_deviation = std::abs(mean);
			Logger::WriteMessage((std::string("Mean deviation = ") + std::to_string(mean_deviation) + "\n").c_str());
			Assert::IsTrue(mean_deviation < Real(1.5e-2), L"Too high deviation from the expected mean value.");
			const auto stdev_deviation = std::abs(std::sqrt(Real(1) / 3) - stdev);
			Logger::WriteMessage((std::string("Stdev deviation = ") + std::to_string(stdev_deviation) + "\n").c_str());
			Assert::IsTrue(stdev_deviation < Real(8.5e-3), L"Too high deviation from the expected StdDev value.");
		}

		TEST_METHOD(RandomVector2dTest)
		{
			std::set<Real> set{};

			const std::size_t iterations = 10000;

			for (auto iter_id = 0; iter_id < iterations; iter_id++)
			{
				const auto random_vector = Vector2d<Real>::random();
				set.emplace(random_vector.x);
				set.emplace(random_vector.y);
			}

			check_random_values(set);

			//In principle, there is a very low probability that during those very few iterations that we did above
			// we will get two identical random values. so we assert this
			Assert::IsTrue(set.size() >= iterations * 2 - 0.0002* iterations, L"Too many identical values.");
		}

		TEST_METHOD(RandomMatrix2x2Test)
		{
			std::set<Real> set{};

			const std::size_t iterations = 10000;

			for (auto iter_id = 0; iter_id < iterations; iter_id++)
			{
				const auto random_matrix = Matrix2x2<Real>::random();
				set.emplace(random_matrix.a00);
				set.emplace(random_matrix.a01);
				set.emplace(random_matrix.a10);
				set.emplace(random_matrix.a11);
			}

			check_random_values(set);

			//In principle, there is a very low probability that during those very few iterations that we did above
			// we will get two identical random values. So we assert this
			Assert::IsTrue(set.size() >= iterations * 4 - 0.0002 * iterations, L"Too many identical values.");
		}

		/// <summary>
		/// Test for consistency of equality/inequality operators
		/// Returns "true" if the two objects are equal and not "not equal" according to their "==" and "!=" operators
		/// </summary>
		template <class TestType>
		bool check_equality(const TestType& obj1, const TestType& obj2) const
		{
			return obj1 == obj2 && !(obj1 != obj2);
		}

		/// <summary>
		/// Test for consistency of equality/inequality operators
		/// Returns "true" if the two objects are "not equal" and not "equal" according to their "==" and "!=" operators
		/// </summary>
		template <class TestType>
		bool check_not_equality(const TestType& obj1, const TestType& obj2) const
		{
			return obj1 != obj2 && !(obj1 == obj2);
		}

		TEST_METHOD(Vector2dEqualityOperatorTest)
		{
			const auto vector = Vector2d<Real>::random();
			const auto vectorCopy = vector;
			const auto term = Utils::get_random(1, 2);//strictly positive random value

			Assert::IsTrue(check_equality(vector, vectorCopy), L"Copy operator does not produce an equal vector.");
			Assert::IsTrue(check_equality(vector, Vector2d<Real>{vector.x, vector.y}), L"The two vectors are not equal!?");
			Assert::IsTrue(check_not_equality(vector, Vector2d<Real>{vector.x + term, vector.y}), L"The two vectors are equal (though their x terms are different)!?");
			Assert::IsTrue(check_not_equality(vector, Vector2d<Real>{vector.x, vector.y + term}), L"The two vectors are equal (though their y terms are different)!?");
		}

		TEST_METHOD(Matrix2x2EqualityOperatorTest)
		{
			const auto matrix = Matrix2x2<Real>::random();
			const auto matrixCopy = matrix;
			const auto term = Utils::get_random(1, 2);//strictly positive random value

			Assert::IsTrue(check_equality(matrix, matrixCopy), L"Copy operator does not produce an equal matrix.");
			Assert::IsTrue(check_equality(matrix, Matrix2x2<Real>{matrix.a00, matrix.a01, matrix.a10, matrix.a11}),
				L"The two matrices are not equal!?");
			Assert::IsTrue(check_not_equality(matrix, Matrix2x2<Real>{matrix.a00 + term, matrix.a01, matrix.a10, matrix.a11}),
				L"The two matrices are equal (though their a00 terms are different)!?");
			Assert::IsTrue(check_not_equality(matrix, Matrix2x2<Real>{matrix.a00, matrix.a01 + term, matrix.a10, matrix.a11}),
				L"The two matrices are equal (though their a01 terms are different)!?");
			Assert::IsTrue(check_not_equality(matrix, Matrix2x2<Real>{matrix.a00, matrix.a01, matrix.a10 + term, matrix.a11}),
				L"The two matrices are equal (though their a10 terms are different)!?");
			Assert::IsTrue(check_not_equality(matrix, Matrix2x2<Real>{matrix.a00, matrix.a01, matrix.a10, matrix.a11 + term}),
				L"The two matrices are equal (though their a11 terms are different)!?");
		}

		TEST_METHOD(Vector2dLInfinityNormTest)
		{
			//Arrange
			const auto vector = Vector2d<Real>::random(); //Vector with random elements within (-1, 1)
			const auto big_term = Utils::get_random(1, 2);//Random value that is greater than the elements in the vector

			//Assert
			Assert::IsTrue(vector.max_abs() >= 0 && vector.max_abs() < Real(1), L"Unexpected value of L-infinity norm.");
			Assert::IsTrue((-vector).max_abs() == vector.max_abs(), L"L-infinity norm of a `minus vector` should be equal to that of the `vector`.");
			Assert::IsTrue(Vector2d<Real>{big_term, vector.y}.max_abs() == big_term &&
				Vector2d<Real>{-big_term, vector.y}.max_abs() == big_term &&
				Vector2d<Real>{vector.x, big_term}.max_abs() == big_term &&
				Vector2d<Real>{vector.x, -big_term}.max_abs() == big_term, L"L-infinity norm treats vector components in a wrong way.");
		}

		TEST_METHOD(Matrix2x2MaxAbsTest)
		{
			//Arrange
			const auto matrix = Matrix2x2<Real>::random(); //Matrix with random elements within (-1, 1)
			const auto big_term = Utils::get_random(1, 2);//Random value that is greater than the elements in the vector

			//Assert
			Assert::IsTrue(matrix.max_abs() >= 0 && matrix.max_abs() < Real(1), L"Unexpected value of max_abs.");
			Assert::IsTrue((-matrix).max_abs() == matrix.max_abs(), L"max_abs of a `minus vector` should be equal to that of the `vector`.");
			Assert::IsTrue(Matrix2x2<Real>{big_term, matrix.a01, matrix.a10, matrix.a11}.max_abs() == big_term &&
				Matrix2x2<Real>{-big_term, matrix.a01, matrix.a10, matrix.a11}.max_abs() == big_term &&
				Matrix2x2<Real>{matrix.a00, big_term, matrix.a10, matrix.a11}.max_abs() == big_term &&
				Matrix2x2<Real>{matrix.a00, -big_term, matrix.a10, matrix.a11}.max_abs() == big_term &&
				Matrix2x2<Real>{matrix.a00, matrix.a01, big_term, matrix.a11}.max_abs() == big_term &&
				Matrix2x2<Real>{matrix.a00, matrix.a01, -big_term, matrix.a11}.max_abs() == big_term &&
				Matrix2x2<Real>{matrix.a00, matrix.a01, matrix.a10, big_term}.max_abs() == big_term &&
				Matrix2x2<Real>{matrix.a00, matrix.a01, matrix.a10, -big_term}.max_abs() == big_term,
				L"max_abs treats vector components in a wrong way.");
		}

		/// <summary>
		/// General method to test properties of zero element
		/// </summary>
		template <class TestType>
		void test_zero_element_properties()
		{
			//Arrange
			const auto zero_element = TestType::zero();
			const auto some_element = TestType::random();
			const TestType default_initialized_element{};//expect to be zero as well

			//Act
			const auto sum_with_zero1 = some_element + zero_element;
			const auto sum_with_zero2 = zero_element + some_element;

			//Assert
			Assert::IsTrue(sum_with_zero1 == some_element && sum_with_zero2 == some_element, L"Adding zero to an element must be equal to the element.");
			Assert::IsTrue(zero_element == default_initialized_element, L"Default initialization is not equalt to zero initialization.");
		}

		/// <summary>
		/// General method to test properties of the inverse element of an additive group
		/// </summary>
		template <class TestType>
		void test_additive_inverse_element()
		{
			//Arrange
			const auto element1 = TestType::random();
			const auto element2 = TestType::random();;
			const auto inverse_element1 = -element1;

			//Act
			const auto result1 = element1 - element1;
			const auto result2 = element1 + inverse_element1;

			const auto result3 = element2 - element1;
			const auto result4 = element2 + (-element1); 

			//Assert
			Assert::IsTrue(result1 == TestType::zero(),
				L"Subtracting a element from itself does not result in zero element.");
			Assert::IsTrue(result2 == TestType::zero(),
				L"Adding an inverse element to the original element does not result in zero element.");

			Assert::IsTrue(result3 == result4,
				L"Subtraction is not equivalent to an addition of inverse.");
		}

		/// <summary>
		/// General method to test commutativity property of the addition operator
		/// </summary>
		template <class TestType>
		void test_addition_commutativity()
		{
			//Arrange
			const auto some_element = TestType::random();
			const auto some_other_element = TestType::random();

			//Act
			const auto result1 = some_element + some_other_element;
			const auto result2 = some_other_element + some_element;

			//Assert
			Assert::IsTrue(result1 == result2, L"Addition operator is not commutative.");
		}

		/// <summary>
		/// General method to test associativity property of the addition operator
		/// </summary>
		template <class R, template<class> class TestType>
		void test_addition_associativity()
		{
			//Arrange
			const auto element1 = TestType<R>::random();
			const auto element2 = TestType<R>::random();
			const auto element3 = TestType<R>::random();

			//Act
			const auto result1 = (element1 + element2) + element3;
			const auto result2 = element1 + (element2 + element3);

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() <= 10 * std::numeric_limits<R>::epsilon() ,
				L"Addition operator is not associative.");
		}

		template <class R, template<class> class TestType>
		void test_distributivity_of_scalar_multiplication_with_respect_to_addition()
		{
			//Arrange
			const auto element1 = TestType<R>::random();
			const auto element2 = TestType<R>::random();
			const auto scalar1 = Utils::get_random(-1, 1);
			const auto scalar2 = Utils::get_random(-1, 1);

			//Act
			const auto result1 = (element1 + element2) * (scalar1 + scalar2);
			const auto result2 = (scalar1 + scalar2) * (element1 + element2);
			const auto result3 = scalar1 * element1 + element2 * scalar1 + element1 * scalar2 + scalar2 * element2;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<R>::epsilon() &&
				(result1 - result3).max_abs() < 10 * std::numeric_limits<R>::epsilon(), L"Distributivity property does not hold true.");
		}

		template <class R, template<class> class TestType>
		void test_distributivity_of_scalar_multiplication_with_respect_to_subtraction()
		{
			//Arrange
			const auto element1 = TestType<R>::random();
			const auto element2 = TestType<R>::random();
			const auto scalar1 = Utils::get_random(-1, 1);
			const auto scalar2 = Utils::get_random(-1, 1);

			//Act
			const auto result1 = (element1 - element2) * (scalar1 - scalar2);
			const auto result2 = (scalar1 - scalar2) * (element1 - element2);
			const auto result3 = scalar1 * element1 - element2 * scalar1 - element1 * scalar2 + scalar2 * element2;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<R>::epsilon() &&
				(result1 - result3).max_abs() < 10 *std::numeric_limits<R>::epsilon(), L"Distributivity property does not hold true.");
		}

		TEST_METHOD(Vector2dZeroElementTest)
		{
			test_zero_element_properties<Vector2d<Real>>();
		}

		TEST_METHOD(Vector2dInverseElementTest)
		{
			test_additive_inverse_element<Vector2d<Real>>();
		}

		TEST_METHOD(Vector2dAdditionCommutativityTest)
		{
			test_addition_commutativity<Vector2d<Real>>();
		}

		TEST_METHOD(Vector2dAdditionAssociativityTest)
		{
			test_addition_associativity<Real, Vector2d>();
		}

		TEST_METHOD(Vector2dDistributivityOfScalarMultiplicationWithRespectToAdditionTest)
		{
			test_distributivity_of_scalar_multiplication_with_respect_to_addition<Real, Vector2d>();
		}

		TEST_METHOD(Vector2dDistributivityOfScalarMultiplicationWithRespectToSubtractionTest)
		{
			test_distributivity_of_scalar_multiplication_with_respect_to_subtraction<Real, Vector2d>();
		}

		TEST_METHOD(Matrix2x2ZeroElementTest)
		{
			test_zero_element_properties<Matrix2x2<Real>>();
		}

		TEST_METHOD(Matrix2x2InverseElementTest)
		{
			test_additive_inverse_element<Matrix2x2<Real>>();
		}

		TEST_METHOD(Matrix2x2AdditionCommputativityTest)
		{
			test_addition_commutativity<Matrix2x2<Real>>();
		}

		TEST_METHOD(Matrix2x2AdditionAssociativityTest)
		{
			test_addition_associativity<Real, Matrix2x2>();
		}

		TEST_METHOD(Matrix2x2DistributivityOfScalarMultiplicationWithRespectToAdditionTest)
		{
			test_distributivity_of_scalar_multiplication_with_respect_to_addition<Real, Matrix2x2>();
		}

		TEST_METHOD(Matrix2x2DistributivityOfScalarMultiplicationWithRespectToSubtractionTest)
		{
			test_distributivity_of_scalar_multiplication_with_respect_to_subtraction<Real, Matrix2x2>();
		}

		TEST_METHOD(DistributivityOfMatruxVectorMultiplicationWithRespectToAdditionTest)
		{
			//Arrange
			const auto vector1 = Vector2d<Real>::random();
			const auto vector2 = Vector2d<Real>::random();
			const auto matrix1 = Matrix2x2<Real>::random();
			const auto matrix2 = Matrix2x2<Real>::random();

			//Act
			const auto result1 = (matrix1 + matrix2) * (vector1 + vector2);
			const auto result2 = (matrix1 * vector1 + matrix2 * vector1 + matrix1 * vector2 + matrix2 * vector2);

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() <= 10 * std::numeric_limits<Real>::epsilon(),
				L"Distributivity does not hold true");
		}

		TEST_METHOD(AssociativityOfMatrixMultiplication)
		{
			//Arrange
			const auto matrix1 = Matrix2x2<Real>::random();
			const auto matrix2 = Matrix2x2<Real>::random();
			const auto matrix3 = Matrix2x2<Real>::random();

			//Act
			const auto result1 = (matrix1 * matrix2) * matrix3;
			const auto result2 = matrix1 * (matrix2 * matrix3);

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() <= 10 * std::numeric_limits<Real>::epsilon(),
				L"Associativity does not hold true");
		}

		TEST_METHOD(DistributivityOfMatrixMulltiplicationWithRespectToAddition)
		{
			//Arrange
			const auto matrix1 = Matrix2x2<Real>::random();
			const auto matrix2 = Matrix2x2<Real>::random();
			const auto matrix3 = Matrix2x2<Real>::random();

			//Act
			const auto result1 = (matrix1 + matrix2) * matrix3;
			const auto result2 = matrix1 * matrix3 + matrix2 * matrix3;

			const auto result3 = matrix3 * (matrix1 + matrix2);
			const auto result4 = matrix3 * matrix1 + matrix3 * matrix2;

			//Assert
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(), L"Distributivity from the right does not hold true");
			Assert::IsTrue((result3 - result4).max_abs() < 10 * std::numeric_limits<Real>::epsilon(), L"Distributivity from the lest does not hold true");
		}

		TEST_METHOD(MatrixProductWithIdentityMatrixTest)
		{
			//Arrange
			const auto matrix = Matrix2x2<Real>::random();
			const auto identity = Matrix2x2<Real>::identity();

			//Act
			const auto result1 = matrix * identity;
			const auto result2 = identity * matrix;

			//Assert
			Assert::IsTrue(result1 == matrix && result2 == matrix, L"The main property of identity matrix does not hold true");
		}

		TEST_METHOD(InverseMatrixTest)
		{
			//Arrange
			const auto matrix = Matrix2x2<Real>::random() + Real(2) * Matrix2x2<Real>::identity();
			const auto identity = Matrix2x2<Real>::identity();
			
			//Act
			const auto result1 = matrix * matrix.inverse();
			const auto result2 = matrix.inverse() * matrix;

			//Assert
			const auto diff1 = (result1 - identity).max_abs();
			const auto diff2 = (result2 - identity).max_abs();
			Assert::IsTrue((result1 - identity).max_abs() <= 10 *std::numeric_limits<Real>::epsilon() &&
				(result2 - identity).max_abs() <= 10 * std::numeric_limits<Real>::epsilon(),
				L"Main property of the inverse matrix does not hold true");
		}

		TEST_METHOD(InversingSingularMatrixResultsInException)
		{
			//Arrange
			const auto singular_matrix = Matrix2x2<Real>{ 1, 2, 3, 6 };
			//Sanity check
			Assert::IsTrue(std::abs(singular_matrix.det()) < 10 * std::numeric_limits<Real>::epsilon(),
				L"The matrix is expected to be singular");

			//Act + Assert
			Assert::ExpectException<std::exception>([&]() {singular_matrix.inverse(); },
				L"Inversing singular matrix must result in an exception thrown");
		}

		TEST_METHOD(TransposeMatrixTest)
		{
			//Arrange
			const auto matrix = Matrix2x2<Real>::random();

			//Act
			const auto transpose_matrix = matrix.transpose();
			const auto double_transpose_matrix = transpose_matrix.transpose();

			//Assert
			Assert::IsTrue(matrix.a00 == transpose_matrix.a00 &&
				matrix.a01 == transpose_matrix.a10 &&
				matrix.a10 == transpose_matrix.a01 &&
				matrix.a11 == transpose_matrix.a11, L"Transpose matrix does not follow the definition");

			Assert::IsTrue(double_transpose_matrix == matrix,
				L"Double transpose matrix must be equal to the original one");
		}

		TEST_METHOD(DeterminantOfProductTest)
		{
			//Arrange
			const auto matrix1 = Matrix2x2<Real>::random();
			const auto matrix2 = Matrix2x2<Real>::random();

			//Act
			const auto det1 = matrix1.det();
			const auto det2 = matrix2.det();
			const auto det1x2 = (matrix1 * matrix2).det();

			//Assert
			Assert::IsTrue(std::abs(det1x2 - det1 * det2) < 10 * std::numeric_limits<Real>::epsilon(),
				L"Determinant of a product is not equal to a product of determinants");
		}

		TEST_METHOD(DeterminantOfIdentityTest)
		{
			//Arrange
			const auto identity = Matrix2x2<Real>::identity();

			//Act
			const auto det = identity.det();

			//Assert
			Assert::IsTrue(std::abs(det - Real(1)) < 10 * std::numeric_limits<Real>::epsilon(),
				L"Determinant of a product is not equal to a product of determinants");
		}

		TEST_METHOD(DeterminantOfTransposeMatrix)
		{
			//Arrange
			const auto matrix = Matrix2x2<Real>::random();

			//Act 
			const auto det1 = matrix.det();
			const auto det2 = matrix.transpose().det();

			//Assert
			Assert::IsTrue(det1 == det2,
				L"Determinant of transpose matrix must be equal to that of the original matrix");
		}

		TEST_METHOD(DotProductLinearityTest)
		{
			//Arrange
			const auto vector1 = Vector2d<Real>::random();
			const auto vector2 = Vector2d<Real>::random();
			const auto vector3 = Vector2d<Real>::random();
			const auto scalar1 = Utils::get_random(-1, 1);
			const auto scalar2 = Utils::get_random(-1, 1);

			//Act
			const auto result1 = (vector1 * scalar1 + scalar2 * vector2).dot(vector3);
			const auto result2 = scalar1 * vector1.dot(vector3) + scalar2 * vector2.dot(vector3);

			//Assert
			Assert::IsTrue(std::abs(result1 - result2) < 10 * std::numeric_limits<Real>::epsilon(),
				L"Linearity property of the dot product does not hold true");
		}

		TEST_METHOD(SymmetricityOfDotProductTest)
		{
			//Arrange
			const auto vector1 = Vector2d<Real>::random();
			const auto vector2 = Vector2d<Real>::random();

			//Act
			const auto result1 = vector1.dot(vector2);
			const auto result2 = vector2.dot(vector1);

			//Assert
			Assert::IsTrue(std::abs(result1 - result2) < 10 * std::numeric_limits<Real>::epsilon(),
				L"Symmetricity property of the dot product does not hold true");
		}

		TEST_METHOD(DotProductWithBasisVectorsTest)
		{
			//Arrange
			const auto vector = Vector2d<Real>::random();

			//Act
			const auto x_coord = vector.dot(Vector2d<Real>::OX());
			const auto y_coord = vector.dot(Vector2d<Real>::OY());

			//Assert
			Assert::IsTrue(x_coord == vector.x, L"Unexpected value of the dot product with OX basis vector");
			Assert::IsTrue(y_coord == vector.y, L"Unexpected value of the dot product with OY basis vector");
		}

		TEST_METHOD(RotationMatrixTest)
		{
			//Arrange
			const auto angle = Utils::get_random(0, Real(2*std::numbers::pi));
			const auto vector = Vector2d<Real>::random();

			//Act
			const auto rotation = Matrix2x2<Real>::rotation(angle);
			const auto rotation_back = Matrix2x2<Real>::rotation(-angle);
			const auto vector_rotated = rotation * vector;

			//Assert
			Assert::IsTrue((rotation * rotation_back - Matrix2x2<Real>::identity()).max_abs() <
				10 * std::numeric_limits<Real>::epsilon(), L"Rotations on opposite angles must cancel each other");

			const auto actual_rotation_angle = std::atan2(vector_rotated.y, vector_rotated.x) - std::atan2(vector.y, vector.x);
			const auto angle_diff = std::abs(actual_rotation_angle - angle);
			Logger::WriteMessage((std::string("Angle difference = ") + Utils::to_string(angle_diff) + "\n").c_str());
			Assert::IsTrue(angle_diff < 10 * std::numeric_limits<Real>::epsilon() ||
				std::abs(angle_diff - Real(2 * std::numbers::pi)) < 10 * std::numeric_limits<Real>::epsilon(),
				L"Unexpected actual rotation angle");
			Assert::IsTrue(std::abs(vector.norm() - vector_rotated.norm()) < 10 * std::numeric_limits<Real>::epsilon(),
				L"distance to the rotation center should not change");
		}

		TEST_METHOD(AffineMatrixMiltiplicationTest)
		{
			//Arrange
			const auto vector = Vector2d<Real>::random();
			const auto affine_matrix1 = MatrixAffine2d<Real>::random();
			const auto affine_matrix2 = MatrixAffine2d<Real>::random();

			//Act
			const auto result1 = (affine_matrix1 * affine_matrix2) * vector;
			const auto result2 = affine_matrix1 * (affine_matrix2 * vector);

			//Assert
			const auto diff = (result1 - result2).max_abs();
			Assert::IsTrue((result1 - result2).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Product of affine transformations must be equivalent to successive application the compound transformations");
		}

		TEST_METHOD(InverseAffineTransformationTest)
		{
			//Arrange
			const auto vector = Vector2d<Real>::random();
			const auto non_singular_matrix = Real(2) * Matrix2x2<Real>::identity() + Matrix2x2<Real> ::random();
			const auto affine_transform = MatrixAffine2d<Real>{ non_singular_matrix, Vector2d<Real>::random()};
			const auto vector_transformed = affine_transform * vector;

			//Act
			const auto affine_transform_inverse = affine_transform.inverse();
			const auto vector_transformer_inverse = affine_transform_inverse * vector_transformed;

			//Assert
			Assert::IsTrue((vector - vector_transformer_inverse).max_abs() <
				10 * std::numeric_limits<Real>::epsilon(), L"Unexpected result of the inverse affine transformation");
		}

		TEST_METHOD(RotationAroundPointTest)
		{
			//Arrange
			const auto vector = Vector2d<Real>::random();
			const auto center = Vector2d<Real>::random() + Vector2d<Real>{ 3.0, 3.0 };
			const auto angle = Utils::get_random(0, Real(std::numbers::pi));
			const auto rotation = MatrixAffine2d<Real>::build_rotation(angle, center);

			//Act
			const auto vector_rotated = rotation * vector;

			//Assert
			const auto radius_vect = vector - center;
			const auto radius_vect_rotated = vector_rotated - center;
			const auto actual_angle = std::acos(radius_vect.normalize().dot(radius_vect_rotated.normalize()));
			const auto angle_diff = std::abs(actual_angle - angle);
			const auto radius_diff = std::abs(radius_vect.norm() - radius_vect_rotated.norm());
			Logger::WriteMessage((std::string("angle_diff = ") + Utils::to_string(angle_diff) + "\n").c_str());
			Logger::WriteMessage((std::string("radius_diff = ") + Utils::to_string(radius_diff) + "\n").c_str());

			Assert::IsTrue(angle_diff < 1000 * std::numeric_limits<Real>::epsilon(),
				L"Too high difference between the actual and expected angles");
			Assert::IsTrue(radius_diff < 10 * std::numeric_limits<Real>::epsilon(),
				L"Distance to the rotation center should not change");
		}
	};
}