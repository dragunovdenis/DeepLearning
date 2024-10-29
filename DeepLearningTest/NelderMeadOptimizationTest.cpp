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
#include <Utilities.h>
#include <Math/Optimization/NelderMeadOptimizer.h>
#include <Image8Bit.h>
#include <MsgPackUtils.h>
#include "StandardTestUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NelderMeadOptimizationTest)
	{
		TEST_METHOD(RegularSimplexConstructionTest)
		{
			//Arrange
			constexpr int N = 5;
			constexpr auto edge_length = static_cast<Real>(1);
			const auto init_point = VectorNdReal<N>(Utils::get_random_std_vector(N, -1, 1));

			//Act
			const auto regular_simplex = NelderMeadOptimizer<N>::create_regular_simplex(init_point, edge_length);

			//Assert
			Assert::IsTrue(regular_simplex[0] == init_point, 
		L"The first vertex in the simplex must coincide with the init. point");
			for (auto i = 0; i < N + 1; ++i)
				for (auto j = i + 1; j < N + 1; ++j)
				{
					const auto edge_length_actual = (regular_simplex[i] - regular_simplex[j]).length();

					Assert::IsTrue(std::abs(edge_length - edge_length_actual) < 10 *std::numeric_limits<Real>::epsilon(),
						L"Unexpected edge length");
				}
		}

		TEST_METHOD(AxesAlignedSimplexConstructionTest)
		{
			//Arrange
			constexpr int N = 5;
			constexpr auto edge_length = static_cast<Real>(1);
			const auto init_point = VectorNdReal<N>(Utils::get_random_std_vector(N, -1, 1));

			//Act
			const auto axes_aligned_simplex = NelderMeadOptimizer<N>::create_axes_aligned_simplex(init_point, edge_length);

			//Assert
			Assert::IsTrue(axes_aligned_simplex[0] == init_point,
				L"The first vertex in the simplex must coincide with the init. point");
			for (auto j = 1; j < N + 1; ++j)
			{
				const auto edge_length_actual = (axes_aligned_simplex[0] - axes_aligned_simplex[j]).length();

				Assert::IsTrue(std::abs(edge_length - edge_length_actual) < 10 * std::numeric_limits<Real>::epsilon(),
					L"Unexpected edge length");
			}

			const auto edge_length_ex = std::sqrt(2) * edge_length;
			for (auto i = 1; i < N + 1; ++i)
				for (auto j = i + 1; j < N + 1; ++j)
				{
					const auto edge_length_actual = (axes_aligned_simplex[i] - axes_aligned_simplex[j]).length();

					Assert::IsTrue(std::abs(edge_length_ex - edge_length_actual) < 10 * std::numeric_limits<Real>::epsilon(),
						L"Unexpected edge length");
				}
		}

		/// <summary>
		/// Returns diagnostics function that allows to track path of "amoeba" in case of 2-dimensional optimization
		/// </summary>
		/// <param name="domain_min_pt">Minimal point of the domain</param>
		/// <param name="domain_max_pt">Maximal point of the domain</param>
		/// <param name="folder_path">Path to a folder to save images</param>
		static NelderMeadOptimizer<2>::DiagnosticsFunc get_diagnostics_func(const VectorNdReal<2>& domain_min_pt,
			const VectorNdReal<2>& domain_max_pt, const std::filesystem::path& folder_path)
		{
			return [domain_min_pt, domain_max_pt, folder_path, counter = 0] (const std::array<VectorNdReal<2>, 3>& simplex,
				const int min_vertex_id, const Real& simplex_size) mutable
			{
				constexpr auto width = 1024;
				constexpr auto height = 1024;
				Image8Bit image(height, width);

				const auto domain_size = domain_max_pt - domain_min_pt;

				for (auto vertex : simplex)
				{
					const auto p_x = static_cast<int>(width * (vertex[0] - domain_min_pt[0]) / domain_size[0]);
					const auto p_y = height - static_cast<int>(height * (vertex[1] - domain_min_pt[1]) / domain_size[1]);

					for (auto i = -3; i < 3; ++i)
					{
						for (auto j = -3; j < 3; ++j)
						{
							const auto p_x_local = p_x + i;
							const auto p_y_local = p_y + j;

							if (p_x_local < 0 || p_x_local >= static_cast<int>(image.width()) ||
								p_y_local < 0 || p_y_local >= static_cast<int>(image.height()))
								continue;

							image(p_y_local, p_x_local) = 255;
						}
					}
				}

				image.SaveToBmp(folder_path / (std::string("image_") + std::to_string(counter) + ".bmp"));
				++counter;
			};
		}

		/// <summary>
		/// General method to run optimization in 2d space
		/// </summary>
		static void Run2dOptimizationTest(const NelderMeadOptimizer<2>::CostFunc& cost_func,
			const VectorNdReal<2>& domain_min_pt, const VectorNdReal<2>& domain_max_point,
			const VectorNdReal<2>& init_point,
			const VectorNdReal<2>& expected_min_point, const Real& expected_min_value,
			const Real& tol_pt = 1e-6, const Real& tol_value = 1e-12)
		{
			NelderMeadOptimizer<2> optimizer;

			optimizer.set_constraints_lower(domain_min_pt);
			optimizer.set_constraints_upper(domain_max_point);
			constexpr auto epsilon = static_cast<Real>(1e-6);
			optimizer.set_min_simplex_size(epsilon);

			//Act
			optimizer.optimize(cost_func, static_cast<Real>(1.1), init_point, false, true
			/*, get_diagnostics_func(domain_min_pt, domain_max_point, "D:\\Temp\\Amoeba")*/);

			//Assert
			const auto min_point = optimizer.get_min_vertex();
			const auto point_diff = (min_point - expected_min_point).max_abs();
			const auto min_value = optimizer.get_min_value();
			const auto value_diff = std::abs(min_value - expected_min_value);
			StandardTestUtils::LogAndAssertLessOrEqualTo(
				"Difference in the position of minimum point found", point_diff, tol_pt);
			StandardTestUtils::LogAndAssertLessOrEqualTo(
				"Difference in the minimum value found", value_diff, tol_value);
		}

		TEST_METHOD(BealeFunctionMinimizationTest)
		{
			//Arrange
			const NelderMeadOptimizer<2>::CostFunc cost_func = [](const VectorNdReal<2>& v)
			{
				return Utils::sqr(static_cast<Real>(1.5) - v[0] + v[0] * v[1]) +
					Utils::sqr(static_cast<Real>(2.25) - v[0] + v[0] * v[1] * v[1]) +
					Utils::sqr(static_cast<Real>(2.625) - v[0] + v[0] * v[1] * v[1] * v[1]);
			};

			Run2dOptimizationTest(cost_func,
				{ static_cast<Real>(-4.5), static_cast<Real>(-4.5)},
				{ static_cast<Real>(4.5), static_cast<Real>(4.5)},
				{ static_cast<Real>(-3.8), static_cast<Real>(3.8)},
				{ static_cast<Real>(3.0), static_cast<Real>(0.5)},
				static_cast<Real>(0.0));
		}

		TEST_METHOD(BoothFunctionMinimizationTest)
		{
			//Arrange
			const NelderMeadOptimizer<2>::CostFunc cost_func = [](const VectorNdReal<2>& v)
			{
				return Utils::sqr(v[0] + 2 * v[1] - 7) + Utils::sqr(2 * v[0] + v[1] - 5);
			};

			Run2dOptimizationTest(cost_func,
				{ -10, -10 }, { 10, 10 }, { -8, -7 }, { 1.0, 3.0 }, 0.0);
		}

		TEST_METHOD(GoldstainPriceFunctionMinimizationTest)
		{
			//Arrange
			const NelderMeadOptimizer<2>::CostFunc cost_func = [](const VectorNdReal<2>& v)
			{
				return (1 + Utils::sqr(v[0] + v[1] + 1) *
					(19 - 14 * v[0] + 3 * v[0] * v[0] - 14 * v[1] + 6 * v[0] * v[1] + 3 * v[1] * v[1]))*
					(30 + Utils::sqr(2 * v[0] - 3 * v[1]) *
					(18 - 32 * v[0] + 12 * v[0] * v[0] + 48 * v[1] - 36 * v[0] * v[1] + 27 * v[1] * v[1]));
			};

			constexpr auto double_pres_calc = std::is_same_v<Real, double>;

			Run2dOptimizationTest(cost_func,
				{ static_cast<Real>(-2), static_cast<Real>(-3) }, 
				{ static_cast<Real>(2), static_cast<Real>(2) }, 
				{ static_cast<Real>(-1.5), static_cast<Real>(-2.5) },
				{ static_cast<Real>(0), static_cast<Real>(-1.0) },
				static_cast<Real>(3.0),
				double_pres_calc ? static_cast<Real>(1e-6) : static_cast<Real>(6e-5),
				double_pres_calc ? static_cast<Real>(1e-11) : static_cast<Real>(5e-5));
		}

		/// <summary>
		/// Returns the Rosenbrock function of the corresponding dimension
		/// </summary>
		template <int N>
		static typename NelderMeadOptimizer<N>::CostFunc get_Rosenbrock_func()
		{
			return [](const VectorNdReal<N>& v)
			{
				auto result = static_cast<Real>(0);

				for (auto i = 0; i < N - 1; ++i)
					result += 100 * Utils::sqr(v[i + 1] - v[i] * v[i]) + Utils::sqr(1 - v[i]);

				return result;
			};
		}

		/// <summary>
		/// General method to run multi-dimensional optimization test
		/// </summary>
		template <int N>
		void RunRosenbrockOptimizationTest(const VectorNdReal<N>& init_pt,
			const Real epsilon = static_cast<Real>(1e-6))
		{
			//Arrange
			NelderMeadOptimizer<N> optimizer;
			optimizer.set_min_simplex_size(epsilon);

			//Act
			optimizer.optimize(get_Rosenbrock_func<N>(), 1, init_pt);

			//Assert
			const auto point_diff = (VectorNdReal<N>(1) - optimizer.get_min_vertex()).max_abs();
			const auto value_diff = std::abs(optimizer.get_min_value());

			Assert::IsTrue(point_diff < epsilon, L"Unexpected point of minimum");
			Assert::IsTrue(value_diff < 500 * std::numeric_limits<Real>::epsilon(), L"Unexpected minimum value");
		}

		TEST_METHOD(Rosenbrock5dOptimizationTest)
		{
			RunRosenbrockOptimizationTest<5>(VectorNdReal<5>(4));
		}

		TEST_METHOD(ParabolaOptimizationTest)
		{
			//Arrange
			NelderMeadOptimizer<1> optimizer;
			constexpr auto epsilon = static_cast<Real>(1e-6);
			optimizer.set_min_simplex_size(epsilon);
			constexpr auto min_pt = static_cast<Real>(3.34);
			const auto cost_func = [](const auto& v)
			{ return 2 * Utils::sqr(v[0] - min_pt); };

			//Act
			optimizer.optimize(cost_func, 1, { -4 });

			//Assert
			const auto point_diff = std::abs(min_pt - optimizer.get_min_vertex()[0]);
			const auto value_diff = std::abs(optimizer.get_min_value());

			Assert::IsTrue(point_diff < epsilon, L"Unexpected point of minimum");
			Assert::IsTrue(value_diff < 500 * std::numeric_limits<Real>::epsilon(), L"Unexpected minimum value");
		}

		TEST_METHOD(PackingTest)
		{
			//Arrange
			constexpr auto N = 7;
			NelderMeadOptimizer<N> optimizer;
			optimizer.set_min_simplex_size(static_cast<Real>(0.1));
			//Do some optimization so that the component has a nontrivial state
			optimizer.optimize(get_Rosenbrock_func<N>(), 1, VectorNdReal<N>(0));

			//Act
			const auto msg = MsgPack::pack(optimizer);
			const auto optimizer_unpacked = MsgPack::unpack<NelderMeadOptimizer<N>>(msg);

			//Assert
			Assert::IsTrue(optimizer.equal_state(optimizer_unpacked), L"De-serialized instance is not equal to the original one.");
		}

	};
}