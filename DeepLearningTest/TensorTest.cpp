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
#include <Math/Tensor.h>
#include "Math/CostFunction.h"
#include <MsgPackUtils.h>
#include "Utilities.h"
#include "StandardTestUtils.h"
#include "Math/ConvolutionUtils.h"
#include "Math/PoolOperator.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(TensorTest)
	{
		/// <summary>
		/// Returns random instances of Tensor class of given dimensions
		/// </summary>
		static Tensor TensorFactory(const std::size_t layer_dim = 7, const std::size_t row_dim = 10, const std::size_t col_dim = 13)
		{
			return Tensor(layer_dim, row_dim, col_dim, -1, 1);
		}

		TEST_METHOD(TensorCopyConstructorTest)
		{
			StandardTestUtils::CopyConstructorTest<Tensor>([]() {return TensorFactory();  });
		}

		TEST_METHOD(TensorCopyAssignmentOperatorTest)
		{
			StandardTestUtils::AssignmentOperatorTest<Tensor>([]() {return TensorFactory(7, 10, 13);  }, []() {return TensorFactory(5, 11, 15); });
		}

		TEST_METHOD(TensorMoveConstructorTest)
		{
			StandardTestUtils::MoveConstructorTest<Tensor>([]() {return TensorFactory();  });
		}

		TEST_METHOD(TensorMoveAssignmentTest)
		{
			StandardTestUtils::MoveAssignmentTest<Tensor>([]() {return TensorFactory();  });
		}

		/// <summary>
		/// Returns random instances of Tensor class of given dimensions
		/// </summary>
		static Tensor TensorFactory(const Index3d& size)
		{
			return TensorFactory(size.x, size.y, size.z);
		}

		TEST_METHOD(TensorPackingTest)
		{
			StandardTestUtils::PackingTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(SumWithZeroTensorTest)
		{
			constexpr auto row_dim = 10;
			constexpr auto col_dim = 13;
			constexpr auto layer_dim = 7;
			StandardTestUtils::SumWithZeroElementTest<Tensor>([]() { return TensorFactory(layer_dim, row_dim, col_dim); }, Tensor(layer_dim, row_dim, col_dim));
		}

		TEST_METHOD(AdditionCommutativityTest)
		{
			StandardTestUtils::AdditionCommutativityTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(DifferenceOfEqualTensorsIsZeroTest)
		{
			StandardTestUtils::DifferenceOfEqualInstancesTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(TensorAdditionAssociativityTest)
		{
			StandardTestUtils::AdditionAssociativityTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(DistributivityOfScalarMultiplicationWuthRespectToTensorAdditionTest)
		{
			StandardTestUtils::ScalarMultiplicationDistributivityTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(TensorMultiplecationByOneTest)
		{
			StandardTestUtils::ScalarMultiplicationByOneTest<Tensor>([]() { return TensorFactory(); });
		}

		TEST_METHOD(ConvolutionNoPaddingNoStrideTest)
		{
			//arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			//Act
			const auto result = tensor.convolve(kernel);
			const auto result_size = result.size_3d();

			//Assert
			Assert::IsTrue(result_size.x == tensor_size.x - kernel_size.x + 1 &&
				result_size.y == tensor_size.y - kernel_size.y + 1 &&
				result_size.z == tensor_size.z - kernel_size.z + 1, L"Wrong size of the convolution result");

			for (auto r_l = 0ll; r_l < result_size.x; r_l++)
			{
				for (auto r_r = 0ll; r_r < result_size.y; r_r++)
				{
					for (auto r_c = 0ll; r_c < result_size.z; r_c++)
					{
						double reference = 0;

						for (auto k_l = 0ll; k_l < kernel_size.x; k_l++)
						{
							for (auto k_r = 0ll; k_r < kernel_size.y; k_r++)
							{
								for (auto k_c = 0ll; k_c < kernel_size.z; k_c++)
								{
									reference += kernel(k_l, k_r, k_c) * tensor(r_l + k_l, r_r + k_r, r_c + k_c);
								}
							}
						}

						const auto diff = std::abs(reference - result(r_l, r_r, r_c));
						Logger::WriteMessage((std::string("diff =  ") + Utils::to_string(diff) + "\n").c_str());
						Assert::IsTrue(diff < 10 * std::numeric_limits<Real>::epsilon(), L"Unexpected result");
					}
				}
			}
		}

		TEST_METHOD(ConvolutionWithPaddingNoStrideTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto padding = Index3d{ 2, 3, 5 };
			//zero tensor
			auto tensor_with_padding = Tensor(tensor_size.x + 2 * padding.x, tensor_size.y + 2 * padding.y, tensor_size.z + 2 * padding.z );

			//copy values to proper location
			for (auto l = 0ll;  l < tensor_size.x; l++ )
				for (auto r = 0ll; r < tensor_size.y; r++)
					for (auto c = 0ll; c < tensor_size.z; c++)
						tensor_with_padding(l + padding.x, r + padding.y, c + padding.z) = tensor(l, r, c);


			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			//Act
			const auto result = tensor.convolve(kernel, padding);
			const auto result_padding_included = tensor_with_padding.convolve(kernel);

			//Assert
			Assert::IsTrue(result == result_padding_included, L"Tensors are not the same");
		}

		TEST_METHOD(ConvolutionWithStrideNoPaddingTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto stride = Index3d{ 2, 3, 5 };

			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			//Act
			const auto result = tensor.convolve(kernel, Index3d{ 0, 0, 0 }, stride);
			const auto result_no_stride = tensor.convolve(kernel);

			//Assert
			const auto result_size = result.size_3d();
			const auto result_no_stride_size = result_no_stride.size_3d();
			Assert::IsTrue((result_no_stride_size.x - 1ull) / stride.x + 1ull == result_size.x &&
						   (result_no_stride_size.y - 1ull) / stride.y + 1ull == result_size.y &&
						   (result_no_stride_size.z - 1ull) / stride.z + 1ull == result_size.z, L"Unexpected size of output tensors");

			for (auto l = 0ul; l < result.layer_dim(); l++)
				for (auto r = 0ul; r < result.row_dim(); r++)
					for (auto c = 0ul; c < result.col_dim(); c++)
					{
						const auto diff = std::abs(result(l, r, c) - result_no_stride(l * stride.x, r * stride.y, c * stride.z));
						Logger::WriteMessage((std::string("diff =  ") + Utils::to_string(diff) + "\n").c_str());
						Assert::IsTrue(diff == static_cast<Real>(0), L"Elements are not the same");
					}
		}

		TEST_METHOD(ConvolutionKernelGradientTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			const auto strides = Index3d{ 2, 1, 5 };
			const auto paddings = Index3d{ 0, 3, 4 };
			const auto conv_res = tensor.convolve(kernel, paddings, strides);
			const auto res_size = conv_res.size_3d();
			const auto reference = Tensor(res_size, true); //zero reference

			const auto cost_func = CostFunction<Tensor>(CostFunctionId::SQUARED_ERROR);

			//Act
			const auto [cost, cost_grad] = cost_func.func_and_deriv(conv_res, reference);
			const auto [kern_grad, in_grad] = tensor.convolution_gradient(cost_grad, kernel, paddings, strides);

			//Assert
			Assert::IsTrue(kern_grad.size_3d() == kernel.size_3d(), L"Unexpected size for the gradient of the convolution kernel.");

			constexpr auto double_precision = std::is_same_v<Real, double>;
			constexpr auto delta = double_precision ? static_cast<Real>(1e-5) : static_cast<Real>(1e-1);

			for (auto k_x = 0ll; k_x < kernel_size.x; k_x++)
				for (auto k_y = 0ll; k_y < kernel_size.y; k_y++)
					for (auto k_z = 0ll; k_z < kernel_size.z; k_z++)
					{
						auto kern_minus_delta = kernel;
						kern_minus_delta(k_x, k_y, k_z) -= delta;

						auto kernel_plus_delts = kernel;
						kernel_plus_delts(k_x, k_y, k_z) += delta;

						const auto conv_res_minus_delta = tensor.convolve(kern_minus_delta, paddings, strides);
						const double cost_minus_delta = cost_func(conv_res_minus_delta, reference);

						const auto conv_res_plus_delta = tensor.convolve(kernel_plus_delts, paddings, strides);
						const double cost_plus_delta = cost_func(conv_res_plus_delta, reference);

						const auto deriv_reference = (cost_plus_delta - cost_minus_delta) / (2 * delta);

						const auto abs_diff = std::abs(deriv_reference - kern_grad(k_x, k_y, k_z));
						const auto rel_diff = std::abs(deriv_reference) > 1 ? abs_diff / std::abs(deriv_reference) : abs_diff;

						Logger::WriteMessage((std::string("Rel. diff. =  ") + Utils::to_string(rel_diff) + "\n").c_str());
						Assert::IsTrue(rel_diff < (double_precision ? static_cast<Real>(5e-7) : static_cast<Real>(2.3e-2)),
							L"Too high deviation from reference.");
					}
		}

		TEST_METHOD(ConvolutionKernelGradientWithScaleTest)
		{
			// Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");
			Tensor in_grad(tensor_size, /*fill zeros*/ true);

			const auto strides = Index3d{ 2, 1, 5 };
			const auto paddings = Index3d{ 0, 3, 4 };
			const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);
			const auto res_gradient = TensorFactory(result_size);//filled with random numbers
			const auto kernel_grad_input_data = TensorFactory(kernel_size);//filled with random numbers;
			auto gradient_container = kernel_grad_input_data;
			const auto gradient_scale_factor = Utils::get_random(-1, 1);

			// Act
			tensor.convolution_gradient<true>(res_gradient.get_handle(), in_grad, gradient_container,
				kernel, paddings, strides, gradient_scale_factor);

			// Assert
			const auto [ref_kernel_grad, ref_in_grad] = tensor.convolution_gradient(res_gradient, kernel, paddings, strides);
			const auto diff = (gradient_container - (kernel_grad_input_data * gradient_scale_factor + ref_kernel_grad)).max_abs();
			Logger::WriteMessage((std::string("Gradient diff. =  ") + Utils::to_string(diff) + "\n").c_str());
			Assert::IsTrue(diff < 200 * std::numeric_limits<Real>::epsilon(),
				L"Too high deviation between actual and reference kernel gradients");

			Assert::IsTrue(in_grad == ref_in_grad, L"Reference and actual input gradients must coincide.");
		}

		TEST_METHOD(ConvolutionInputGradientTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 11, 9 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			const auto strides = Index3d{ 2, 1, 5 };
			const auto paddings = Index3d{ 0, 3, 4 };
			const auto conv_res = tensor.convolve(kernel, paddings, strides);
			const auto res_size = conv_res.size_3d();
			const auto reference = Tensor(res_size, true); //zero reference

			const auto cost_func = CostFunction<Tensor>(CostFunctionId::SQUARED_ERROR);

			//Act
			const auto [cost, cost_grad] = cost_func.func_and_deriv(conv_res, reference);
			const auto [kern_grad, in_grad] = tensor.convolution_gradient(cost_grad, kernel, paddings, strides);

			//Assert
			Assert::IsTrue(kern_grad.size_3d() == kernel.size_3d(), L"Unexpected size for the gradient of the convolution kernel.");

			constexpr auto double_precision = std::is_same_v<Real, double>;
			constexpr auto delta = double_precision ? static_cast<Real>(1e-5) : static_cast<Real>(1e-1);

			for (auto t_x = 0ll; t_x < tensor_size.x; t_x++)
				for (auto t_y = 0ll; t_y < tensor_size.y; t_y++)
					for (auto t_z = 0ll; t_z < tensor_size.z; t_z++)
					{
						auto tensor_minus_delta = tensor;
						tensor_minus_delta(t_x, t_y, t_z) -= delta;

						auto tensor_plus_delta = tensor;
						tensor_plus_delta(t_x, t_y, t_z) += delta;

						const auto conv_res_minus_delta = tensor_minus_delta.convolve(kernel, paddings, strides);
						const double cost_minus_delta = cost_func(conv_res_minus_delta, reference);

						const auto conv_res_plus_delta = tensor_plus_delta.convolve(kernel, paddings, strides);
						const double cost_plus_delta = cost_func(conv_res_plus_delta, reference);

						const auto deriv_reference = (cost_plus_delta - cost_minus_delta) / (2 * delta);

						const auto abs_diff = std::abs(deriv_reference - in_grad(t_x, t_y, t_z));
						const auto rel_diff = std::abs(deriv_reference) > 1 ?	abs_diff / std::abs(deriv_reference) : abs_diff;

						Logger::WriteMessage((std::string("Rel. diff. =  ") + Utils::to_string(rel_diff) + "\n").c_str());
						Assert::IsTrue(rel_diff < (double_precision ? static_cast<Real>(2e-8) : static_cast<Real>(1e-3)),
							L"Too high deviation from reference.");
					}
		}

		TEST_METHOD(PoolTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 11, 9 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			const auto strides = Index3d{ 2, 1, 5 };
			const auto paddings = Index3d{ 0, 3, 4 };
			const auto conv_res = tensor.convolve(kernel, paddings, strides);

			const auto pool_operator = KernelPool(kernel);

			//Act
			const auto pool_res = tensor.pool(pool_operator, paddings, strides);

			//Assert
			Assert::IsTrue(conv_res == pool_res, L"Results of convolution and pool operations (with kernel pool operator) must coincide.");
		}

		TEST_METHOD(PoolGradientTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 11, 9 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			const auto kernel_size = Index3d{ 3, 5, 7 };
			const auto kernel = TensorFactory(kernel_size);//filled with random numbers
			Assert::IsTrue(kernel.max_abs() > 0, L"Kernel is expected to be nonzero");

			const auto strides = Index3d{ 2, 1, 5 };
			const auto paddings = Index3d{ 0, 3, 4 };
			const auto conv_res = tensor.convolve(kernel, paddings, strides);
			const auto res_size = conv_res.size_3d();
			const auto reference = Tensor(res_size, true); //zero reference

			const auto cost_func = CostFunction<Tensor>(CostFunctionId::SQUARED_ERROR);
			const auto [cost, cost_grad] = cost_func.func_and_deriv(conv_res, reference);
			const auto [kern_grad, in_grad_ref] = tensor.convolution_gradient(cost_grad, kernel, paddings, strides);

			const auto pool_operator = KernelPool(kernel);

			//Act
			const auto pool_grad = tensor.pool_input_gradient(cost_grad, pool_operator, paddings, strides);

			//Assert
			Assert::IsTrue(in_grad_ref == pool_grad, L"Gradients must coincide.");
		}

		TEST_METHOD(MaxPoolTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 11, 9 };
			const auto max_item_id = Index3d{ Utils::get_random_int(0, static_cast<int>(tensor_size.x - 1)),
											  Utils::get_random_int(0, static_cast<int>(tensor_size.y - 1)),
											  Utils::get_random_int(0, static_cast<int>(tensor_size.z - 1)) };

			const auto max_item_value = Utils::get_random(10, 100);

			auto pool_input_grad_ref = Tensor(tensor_size, true /*assign zeros*/);
			pool_input_grad_ref(max_item_id.x, max_item_id.y, max_item_id.z) = Real(1);
			const auto tensor = TensorFactory(tensor_size) + pool_input_grad_ref * max_item_value;
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");
			const auto strides = Index3d{ 1, 1, 1 };
			const auto paddings = Index3d{ 0, 0, 0 };

			auto pool_res_gradient = Tensor(1, 1, 1);
			pool_res_gradient(0, 0, 0) = Real(1);

			//Create pool operator with the "window" of the size of the tensor
			//(to do the "global" pooling)
			const auto max_pool_operator = MaxPool(tensor.size_3d());

			//Act
			const auto pool_result = tensor.pool(max_pool_operator);
			const auto pool_input_grad = tensor.pool_input_gradient(pool_res_gradient, max_pool_operator, paddings, strides);

			//Assert
			Assert::IsTrue(pool_result.size_3d() == Index3d{ 1, 1, 1 }, L"Unexpected size of the pooling result");
			Assert::IsTrue(pool_input_grad.size_3d() == tensor_size, L"Unexpected size of the pool input gradient");
			Assert::IsTrue(pool_result(0, 0, 0) == tensor(max_item_id.x, max_item_id.y, max_item_id.z), L"Unexpected result of the max-pool operation");
			Assert::IsTrue(pool_input_grad_ref == pool_input_grad, L"Unexpected value of the pool input gradient");
		}

		TEST_METHOD(LayerHandleTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			auto tensor = TensorFactory(tensor_size);//filled with random numbers
			const auto const_tensor = tensor;
			const auto expected_layer_size = tensor.row_dim() * tensor.col_dim();
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");

			//Act + Assert
			for (auto layer_id = 0ull; layer_id < tensor.layer_dim(); layer_id++)
			{
				const auto const_layer_handle = tensor.get_layer_handle(layer_id);
				auto layer_handle = tensor.get_layer_handle(layer_id);
				auto layer_handle_const = const_tensor.get_layer_handle(layer_id);

				Assert::IsTrue(layer_handle.size() == layer_handle_const.size() &&
					layer_handle.size() == expected_layer_size, L"Unexpected size of the layer handles");

				for (auto in_layer_id = 0ull; in_layer_id < layer_handle.size(); in_layer_id++)
				{
					const auto global_id = layer_id * expected_layer_size + in_layer_id;
					const auto expected_value = layer_handle[in_layer_id];
					Assert::IsTrue(const_layer_handle[in_layer_id] == expected_value &&
						layer_handle_const[in_layer_id] == expected_value &&
						const_tensor[global_id] == expected_value,
						L"The handles are expected to point to the memory with the same data");

					//Modify data and check that the corresponding element was modified in the original tensor
					const auto random_value = Utils::get_random(-1, 1);
					layer_handle[in_layer_id] = random_value;
					Assert::IsTrue(tensor[global_id] == random_value, L"Handle does not point to the memory of the original tensor");
				}
			}
		}

		TEST_METHOD(AddScaledTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor1 = TensorFactory(tensor_size);//filled with random numbers
			const auto tensor2 = TensorFactory(tensor_size);//filled with random numbers
			const auto scalar = Utils::get_random(1, 5);
			Assert::IsTrue(tensor1.max_abs() > 0 && tensor2.max_abs() > 0,
				L"Tensors are supposed to be nonzero");

			//Act
			auto result1 = tensor1;
			result1.add_scaled(tensor2, scalar);
			const auto result2 = tensor1 + tensor2 * scalar;

			//Assert
			Assert::IsTrue(result1 == result2, L"Results are supposed to be the same");
		}

		TEST_METHOD(ScaleAndAddTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor1 = TensorFactory(tensor_size);//filled with random numbers
			const auto tensor2 = TensorFactory(tensor_size);//filled with random numbers
			const auto scalar = Utils::get_random(1, 5);
			Assert::IsTrue(tensor1.max_abs() > 0 && tensor2.max_abs() > 0,
				L"Tensors are supposed to be nonzero");

			//Act
			auto result1 = tensor1;
			result1.scale_and_add(tensor2, scalar);
			const auto result2 = tensor1 * scalar  + tensor2;

			//Assert
			Assert::IsTrue(result1 == result2, L"Results are supposed to be the same");
		}

		TEST_METHOD(ScaleAndAddScaledTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 22, 33 };
			const auto tensor1 = TensorFactory(tensor_size);//filled with random numbers
			const auto tensor2 = TensorFactory(tensor_size);//filled with random numbers
			const auto scalar_0 = Utils::get_random(1, 5);
			const auto scalar_1 = Utils::get_random(1, 5);
			Assert::IsTrue(tensor1.max_abs() > 0 && tensor2.max_abs() > 0,
				L"Tensors are supposed to be nonzero");

			//Act
			auto result1 = tensor1;
			result1.scale_and_add_scaled(scalar_0, tensor2, scalar_1);
			const auto result2 = tensor1 * scalar_0 + tensor2 * scalar_1;

			//Assert
			Assert::IsTrue(result1 == result2, L"Results are supposed to be the same");
		}

		void min_max_pool_test_general(const bool max)
		{
			//Arrange
			const auto tensor_size = Index3d{ 10, 11, 9 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");
			const auto pool_window_size = Index3d(2, 3, 4);

			//Calculate reference
			const auto kernel_size = pool_window_size;
			const auto strides = kernel_size;
			const auto paddings = Index3d{ 0, 0, 0 };
			const auto pool_operator = max ? MaxPool(kernel_size).clone() : MinPool(kernel_size).clone();
			//pool result reference
			const auto pool_res_reference = tensor.pool(*pool_operator, paddings, strides);

			const auto res_grad = Tensor(pool_res_reference.size_3d(), static_cast<Real>(-1), static_cast<Real>(1));
			Assert::IsTrue(res_grad.max_abs() > static_cast<Real>(0), L"Result gradient is expected to be nonzero");

			//pool input gradient reference
			const auto input_gradient_reference = tensor.pool_input_gradient(res_grad, *pool_operator, paddings, strides);

			//Act
			const auto [pool_res, out_to_in_mapping] = tensor.min_max_pool(pool_window_size, max);
			const auto input_gradient = tensor.min_max_pool_input_gradient(res_grad, out_to_in_mapping);

			//Assert
			Assert::IsTrue(pool_res == pool_res_reference, L"Actual and expected values of the max pool result are different");
			Assert::IsTrue(input_gradient == input_gradient_reference, L"Actual and expected values of the max pool input gradient are different");
		}

		TEST_METHOD(OptimizedMaxPoolTest)
		{
			min_max_pool_test_general(true /*max*/);
		}

		TEST_METHOD(OptimizedMinPoolTest)
		{
			min_max_pool_test_general(false /*min*/);
		}

		TEST_METHOD(AveragePoolTest)
		{
			//Arrange
			const auto tensor_size = Index3d{ 11, 13, 22 };
			const auto tensor = TensorFactory(tensor_size);//filled with random numbers
			Assert::IsTrue(tensor.max_abs() > 0, L"Tensor is expected to be nonzero");
			const auto pool_window_size = Index3d(2, 3, 4);

			//Calculate reference
			const auto pool_operator = AveragePool(pool_window_size);
			const auto result_reference = tensor.pool(pool_operator, Index3d{ 0 }, pool_window_size);

			const auto res_grad = Tensor(result_reference.size_3d(), static_cast<Real>(-1), static_cast<Real>(1));
			Assert::IsTrue(res_grad.max_abs() > static_cast<Real>(0), L"Result gradient is expected to be nonzero");

			//pool input gradient reference
			const auto input_gradient_reference = tensor.pool_input_gradient(res_grad, pool_operator, Index3d{ 0 }, pool_window_size);

			//Act
			const auto result = tensor.average_pool(pool_window_size);
			const auto input_gradient = tensor.average_pool_input_gradient(res_grad, pool_window_size);

			//Assert
			const auto res_diff = (result_reference - result).max_abs();
			const auto grad_diff = (input_gradient - input_gradient_reference).max_abs();

			Logger::WriteMessage((std::string("Result difference = ") + Utils::to_string(res_diff) + "\n").c_str());
			Logger::WriteMessage((std::string("Gradient difference = ") + Utils::to_string(grad_diff) + "\n").c_str());
			Assert::IsTrue(res_diff <= std::numeric_limits<Real>::epsilon(), L"Too high deviation from the pool reference result");
			Assert::IsTrue(grad_diff <= std::numeric_limits<Real>::epsilon(), L"Too high deviation from the pool reference gradient");
		}
	};
}
