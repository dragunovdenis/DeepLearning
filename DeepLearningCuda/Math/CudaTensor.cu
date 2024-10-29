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

#include "CudaVector.cuh"
#include "CudaMatrix.cuh"
#include "CudaTensor.cuh"
#include <cuda_runtime.h>
#include "CudaUtils.h"
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include <thrust/equal.h>
#include <thrust/scatter.h>
#include <Math/ConvolutionUtils.h>
#include "CudaSetup.h"
#include <Math/Functions.h>

namespace DeepLearning
{
	void CudaTensor::abandon_resources()
	{
		Base::abandon_resources();
		_layer_dim = 0;
		_row_dim = 0;
		_col_dim = 0;
	}

	void CudaTensor::assign(const Tensor& source)
	{
		resize(source.layer_dim(), source.row_dim(), source.col_dim());
		CudaUtils::cuda_copy_host2device(begin(), source.begin(), size());
	}

	void CudaTensor::assign(const CudaTensor& source)
	{
		resize(source.layer_dim(), source.row_dim(), source.col_dim());
		CudaUtils::cuda_copy_device2device(begin(), source.begin(), size());
	}

	void CudaTensor::resize(const Index3d& size_3d)
	{
		resize(size_3d.x, size_3d.y, size_3d.z);
	}

	void CudaTensor::resize(const std::size_t& new_layer_dim, const std::size_t& new_row_dim, const std::size_t& new_col_dim)
	{
		const auto new_size = new_layer_dim * new_row_dim * new_col_dim;
		allocate(new_size);

		_layer_dim = new_layer_dim;
		_row_dim = new_row_dim;
		_col_dim = new_col_dim;
	}

	CudaTensor& CudaTensor::get_resized(const Index3d& size_3d)
	{
		resize(size_3d);
		return *this;
	}

	std::size_t CudaTensor::size() const
	{
		return _layer_dim * _row_dim * _col_dim;
	}

	Tensor CudaTensor::to_host() const
	{
		Tensor result(_layer_dim, _row_dim, _col_dim, false /*assign zero*/);
		CudaUtils::cuda_copy_device2host(result.begin(), begin(), size());

		return result;
	}

	void CudaTensor::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		Tensor proxy;
		msgpack::type::make_define_array(proxy).msgpack_unpack(msgpack_o);
		assign(proxy);
	}

	CudaTensor::CudaTensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const bool assign_zero)
	{
		resize(layer_dim, row_dim, col_dim);

		if (assign_zero) fill_zero();
	}

	CudaTensor::CudaTensor(const Index3d& size, const bool assign_zero) :
		CudaTensor(size.x, size.y, size.z, assign_zero)
	{}

	CudaTensor::CudaTensor(const CudaTensor& tensor) :
		CudaTensor(tensor.layer_dim(), tensor.row_dim(), tensor.col_dim(), false /*assign zero*/)
	{
		CudaUtils::cuda_copy_device2device(begin(), tensor.begin(), CudaTensor::size());
	}

	CudaTensor::CudaTensor(CudaVector&& vector) noexcept :
		_layer_dim(1), _row_dim(1), _col_dim(vector.dim())
	{
		take_over_resources(std::move(vector));
	}

	CudaTensor::CudaTensor(CudaMatrix&& matrix) noexcept :
		_layer_dim(1), _row_dim(matrix.row_dim()), _col_dim(matrix.col_dim())
	{
		take_over_resources(std::move(matrix));
	}

	CudaTensor::CudaTensor(CudaTensor&& tensor) noexcept :
		_layer_dim(tensor.layer_dim()), _row_dim(tensor.row_dim()), _col_dim(tensor.col_dim())
	{
		take_over_resources(std::move(tensor));
	}

	CudaTensor::CudaTensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const Real range_begin, const Real range_end, std::mt19937* seeder) :
		CudaTensor(layer_dim, row_dim, col_dim, false /*assign zero*/)
	{
		uniform_random_fill(range_begin, range_end, seeder);
	}

	CudaTensor::CudaTensor(const Index3d& size, const Real range_begin,
		const Real range_end, std::mt19937* seeder) :
		CudaTensor(size.x, size.y, size.z, range_begin, range_end, seeder)
	{}

	CudaTensor::CudaTensor(const Tensor& source)
	{
		assign(source);
	}

	CudaTensor& CudaTensor::operator =(const CudaTensor& tensor)
	{
		if (this != &tensor)
			assign(tensor);

		return *this;
	}

	CudaTensor& CudaTensor::operator =(CudaVector&& vector) noexcept
	{
		_layer_dim = 1ull;
		_row_dim = 1ull;
		_col_dim = vector.dim();
		take_over_resources(std::move(vector));

		return *this;
	}

	CudaTensor& CudaTensor::operator =(CudaMatrix&& matrix) noexcept
	{
		_layer_dim = 1ull;
		_row_dim = matrix.row_dim();
		_col_dim = matrix.col_dim();
		take_over_resources(std::move(matrix));

		return *this;
	}

	CudaTensor& CudaTensor::operator =(CudaTensor&& tensor) noexcept
	{
		if (this != &tensor)
		{
			_layer_dim = tensor.layer_dim();
			_row_dim = tensor.row_dim();
			_col_dim = tensor.col_dim();
			take_over_resources(std::move(tensor));
		}

		return *this;
	}

	std::size_t CudaTensor::layer_dim() const
	{
		return _layer_dim;
	}

	std::size_t CudaTensor::row_dim() const
	{
		return _row_dim;
	}

	std::size_t CudaTensor::col_dim() const
	{
		return _col_dim;
	}

	CudaTensor& CudaTensor::operator +=(const CudaTensor& tensor)
	{
		if (_layer_dim != tensor._layer_dim ||
			_row_dim != tensor._row_dim ||
			_col_dim != tensor._col_dim)
			throw std::exception("Invalid input");

		add(tensor);

		return *this;
	}

	CudaTensor& CudaTensor::operator -=(const CudaTensor& tensor)
	{
		if (_layer_dim != tensor._layer_dim ||
			_row_dim != tensor._row_dim ||
			_col_dim != tensor._col_dim)
			throw std::exception("Invalid input");

		sub(tensor);

		return *this;
	}

	CudaTensor& CudaTensor::operator *=(const Real& scalar)
	{
		mul(scalar);

		return *this;
	}

	CudaTensor operator +(const CudaTensor& tensor1, const CudaTensor& tensor2)
	{
		auto result = tensor1;
		return result += tensor2;
	}

	CudaTensor operator -(const CudaTensor& tensor1, const CudaTensor& tensor2)
	{
		auto result = tensor1;
		return result -= tensor2;
	}

	CudaTensor operator *(const CudaTensor& tensor, const Real& scalar)
	{
		auto result = tensor;
		return result *= scalar;
	}

	CudaTensor operator *(const Real& scalar, const CudaTensor& tensor)
	{
		return tensor * scalar;
	}

	bool CudaTensor::operator ==(const CudaTensor& tensor) const
	{
		return  layer_dim() == tensor.layer_dim() &&
			    row_dim() == tensor.row_dim() &&
			    col_dim() == tensor.col_dim() &&
			    thrust::equal(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), tensor.begin());
	}

	bool CudaTensor::operator !=(const CudaTensor& tensor) const
	{
		return !(*this == tensor);
	}

	Index3d CudaTensor::size_3d() const
	{
		return { static_cast<long long>(layer_dim()),
			     static_cast<long long>(row_dim()),
			     static_cast<long long>(col_dim()) };
	}

	CudaTensor& CudaTensor::reshape(const Index3d& new_shape)
	{
		if (size() != new_shape.x * new_shape.y * new_shape.z)
			throw std::exception("Invalid shape for the current tensor");

		_layer_dim = new_shape.x;
		_row_dim = new_shape.y;
		_col_dim = new_shape.z;

		return *this;
	}

	template <class T>
	__device__  T coords_to_data_id(const T& layer_id, const T& row_id, const T& col_id,
		const T& row_dim, const T& col_dim)
	{
		return col_dim * (layer_id * row_dim + row_id) + col_id;
	}

	/// <summary>
	/// CUDA kernel to perform convolution operation
	/// </summary>
	/// <param name="tensor">The input tensor to apply convolution to</param>
	/// <param name="tensor_size">Size of the input tensor</param>
	/// <param name="kernel">Convolution kernel tensor</param>
	/// <param name="kernel_size">Size of the convolution kernel tensor</param>
	/// <param name="paddings">Zero paddings to be applied to the input kernel when convolving </param>
	/// <param name="strides">Convolution strides (shifts of the convolution window)</param>
	/// <param name="result">Placeholder for the result of the convolution (tensor)</param>
	/// <param name="result_size">Size of the convolution result (tensor)</param>
	__global__ void convolve_kernel(const Real* __restrict__ tensor, const Index3d tensor_size,
							   const Real* __restrict__ kernel, const Index3d kernel_size,
							   const Index3d paddings, const Index3d strides,
		                       Real* __restrict__ result, const Index3d result_size)
	{
		const auto result_flattened_id = threadIdx.x + blockIdx.x * blockDim.x;
		const auto result_flattened_size = result_size.coord_prod();

		if (result_flattened_id >= result_flattened_size)
			return;

		const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(result_flattened_id, result_size);
		const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
			ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

		Real part_res = Real(0);

		KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
			part_res += tensor[coords_to_data_id(t_x, t_y, t_z, tensor_size.y, tensor_size.z)] *
			kernel[coords_to_data_id(k_x, k_y, k_z, kernel_size.y, kernel_size.z)];)

		result[result_flattened_id] = part_res;
	}

	/// <summary>
	/// CUDA kernel to perform convolution operation; version that uses shared memory
	/// </summary>
	/// <param name="tensor">The input tensor to apply convolution to</param>
	/// <param name="tensor_size">Size of the input tensor</param>
	/// <param name="kernel">Convolution kernel tensor</param>
	/// <param name="kernel_size">Size of the convolution kernel tensor</param>
	/// <param name="paddings">Zero paddings to be applied to the input kernel when convolving </param>
	/// <param name="strides">Convolution strides (shifts of the convolution window)</param>
	/// <param name="result">Placeholder for the result of the convolution (tensor)</param>
	/// <param name="result_size">Size of the convolution result (tensor)</param>
	__global__ void convolve_shared_kernel(const Real* __restrict__ tensor, const Index3d tensor_size,
		const Real* __restrict__ kernel, const Index3d kernel_size,
		const Index3d paddings, const Index3d strides,
		Real* __restrict__ result, const Index3d result_size)
	{
		const auto result_flattened_id = threadIdx.x + blockIdx.x * blockDim.x;
		const auto result_flattened_size = result_size.coord_prod();
		const auto kernel_flattened_size = kernel_size.coord_prod();

		extern __shared__ Real kernel_shared[];

		const auto items_per_thread = Func::cuda_max(10ll, (kernel_flattened_size + blockDim.x - 1) / blockDim.x);
		const auto element_start_id = items_per_thread * threadIdx.x;
		const auto element_stop_id = Func::cuda_min(kernel_flattened_size, element_start_id + items_per_thread);

		for (auto element_id = element_start_id; element_id < element_stop_id; element_id++)
			kernel_shared[element_id] = kernel[element_id];

		__syncthreads();

		if (result_flattened_id >= result_flattened_size)
			return;

		const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(result_flattened_id, result_size);
		const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
			ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

		Real part_res = Real(0);

		KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
			part_res += tensor[coords_to_data_id(t_x, t_y, t_z, tensor_size.y, tensor_size.z)] *
			kernel_shared[coords_to_data_id(k_x, k_y, k_z, kernel_size.y, kernel_size.z)];)

			result[result_flattened_id] = part_res;
	}

	Index3d CudaTensor::convolve(RealMemHandle result_handle, const CudaTensor& kernel, const Index3d& paddings, const Index3d& strides) const
	{
		const auto kernel_size = kernel.size_3d();
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (result_handle.size() != result_size.coord_prod())
			throw std::exception("Unexpected amount of memory to store the result");

		const auto blocks_cnt = CudaSetup::calc_blocks(result_handle.size());

		convolve_kernel<<<blocks_cnt , CudaSetup::max_threads_per_block(), 0, cudaStreamPerThread>>>(begin(), tensor_size,
			kernel.begin(), kernel_size, paddings, strides,	result_handle.data(), result_size);
		CUDA_SANITY_CHECK
		return result_size;
	}

	void CudaTensor::convolve(CudaTensor& result, const std::vector<CudaTensor>& kernels, const Index3d& paddings, const Index3d& strides) const
	{
		if (result.layer_dim() != kernels.size() || kernels.empty())
			throw std::exception("Inconsistent input");

		//It is assumed that all the kernels in the collection have the same size
		const auto kernel_size = kernels[0].size_3d();
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (result.size_3d().yz() != result_size.yz() || result_size.x != 1)
			throw std::exception("Unexpected amount of memory to store the result");

		const auto blocks_cnt = CudaSetup::calc_blocks(result_size.coord_prod());
		const auto kernel_size_bytes = sizeof(Real) * kernel_size.coord_prod();

		for (auto kernel_id = 0ull; kernel_id < kernels.size(); kernel_id++)
		{
			convolve_kernel << <blocks_cnt, CudaSetup::max_threads_per_block(), 0, cudaStreamPerThread>> > (begin(), tensor_size,
				kernels[kernel_id].begin(), kernel_size, paddings, strides,	result.get_layer_handle(kernel_id).data(), result_size);
			CUDA_SANITY_CHECK
		}
	}

	CudaTensor CudaTensor::convolve(const CudaTensor& kernel, const Index3d& paddings, const Index3d& strides) const
	{
		const auto result_dim = ConvolutionUtils::calc_conv_res_size(size_3d(), kernel.size_3d(), paddings, strides);
		auto result = CudaTensor(result_dim, false);

		convolve(result.get_handle(), kernel, paddings, strides);

		return result;
	}

	std::tuple<CudaTensor, CudaTensor> CudaTensor::convolution_gradient(const CudaTensor& conv_res_grad, const CudaTensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		if (ConvolutionUtils::calc_conv_res_size(size_3d(), kernel.size_3d(), paddings, strides) != conv_res_grad.size_3d())
			throw std::exception("Inconsistent input data.");

		auto input_grad = CudaTensor(size_3d(), true);
		const auto kernel_grad = convolution_gradient(conv_res_grad.get_handle(), input_grad, kernel, paddings, strides);

		return std::make_tuple(kernel_grad, input_grad);
	}

	/// <summary>
	/// CUDA kernel to compute gradients of the convolution operation with respect to the input tensor and the kernel
	/// </summary>
	/// <param name="tensor">The input tensor that took part in the convolution operation</param>
	/// <param name="tensor_size">3d size of the input tensor</param>
	/// <param name="res_grad">Tensor-gradient calculated with respect to the output of the convolution operation</param>
	/// <param name="res_grad_size">Size of the output of the convolution operation</param>
	/// <param name="kernel">Kernel that took part in the convolution operation</param>
	/// <param name="kernel_size">3d size of the kernel-tensor</param>
	/// <param name="paddings">Zero-paddings used in the convolution operation</param>
	/// <param name="strides">Strides used in the convolution operation</param>
	/// <param name="input_grad">Container to hold gradient with respect to the input tensor of the convolution (with the dimensions equal to those of the input tensor )</param>
	/// <param name="kernel_grad">Container to hold gradient with respect to the kernel of the convolution (with the dimensions equal to those of the kernel tensor)</param>
	template <bool CALC_INPUT_GRAD>
	__global__ void convolution_gradient_kernel(const Real* __restrict__ tensor, const Index3d tensor_size,
		const Real* __restrict__ res_grad, const Index3d res_grad_size,
		const Real* __restrict__ kernel, const Index3d kernel_size,
		const Index3d paddings, const Index3d strides, Real* __restrict__ input_grad, Real* __restrict__ kernel_grad)
	{
		const auto res_grad_flattened_id = threadIdx.x + blockIdx.x * blockDim.x;
		const auto res_grad_flattened_size = res_grad_size.coord_prod();

		if (res_grad_flattened_id >= res_grad_flattened_size)
			return;

		const auto factor = res_grad[res_grad_flattened_id];
		if (factor == Real(0))
			return;

		const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_grad_flattened_id, res_grad_size);
		const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
			ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

		KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
			const int tensor_data_id = coords_to_data_id(t_x, t_y, t_z, tensor_size.y, tensor_size.z);
		    const int kernel_data_id = coords_to_data_id(k_x, k_y, k_z, kernel_size.y, kernel_size.z);
			if (CALC_INPUT_GRAD)
				atomicAdd(input_grad + tensor_data_id, kernel[kernel_data_id] * factor);

			atomicAdd(kernel_grad + kernel_data_id, tensor[tensor_data_id] * factor);)
	}

	template <bool CALC_INPUT_GRAD>
	CudaTensor CudaTensor::convolution_gradient(const RealMemHandleConst& conv_res_grad, CudaTensor& input_grad, const CudaTensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		auto kernel_grad = CudaTensor(kernel.size_3d(), false);
		convolution_gradient<CALC_INPUT_GRAD>(conv_res_grad, input_grad, kernel_grad, kernel, paddings, strides, static_cast<Real>(0));

		return kernel_grad;
	}

	template <bool CALC_INPUT_GRAD>
	void CudaTensor::convolution_gradient(const RealMemHandleConst& conv_res_grad, CudaTensor& input_grad,
		CudaTensor& kernel_grad, const CudaTensor& kernel, const Index3d& paddings,
		const Index3d& strides, const Real kernel_grad_scale) const
	{
		const auto tensor_size = size_3d();
		const auto kernel_size = kernel.size_3d();
		const auto conv_result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (conv_res_grad.size() != conv_result_size.x * conv_result_size.y * conv_result_size.z)
			throw std::exception("Unexpected size of the convolution result gradient");

		if (CALC_INPUT_GRAD && input_grad.size_3d() != tensor_size)
			throw std::exception("Unexpected size of the input gradient container");

		if (kernel_grad_scale != static_cast<Real>(0))
			kernel_grad *= kernel_grad_scale;
		else
			kernel_grad.fill_zero();

		const auto blocks_cnt = CudaSetup::calc_blocks(conv_res_grad.size());

		convolution_gradient_kernel<CALC_INPUT_GRAD> << <blocks_cnt, CudaSetup::max_threads_per_block(), 0, cudaStreamPerThread >> >
			(begin(), tensor_size, conv_res_grad.data(), conv_result_size, kernel.begin(), kernel_size,
				paddings, strides, input_grad.begin(), kernel_grad.begin());
		CUDA_SANITY_CHECK
	}

	template void CudaTensor::convolution_gradient<true>(const RealMemHandleConst& conv_res_grad, CudaTensor& input_grad,
		CudaTensor& kernel_grad, const CudaTensor& kernel, const Index3d& paddings,
		const Index3d& strides, const Real kernel_grad_scale) const;
	template void CudaTensor::convolution_gradient<false>(const RealMemHandleConst& conv_res_grad, CudaTensor& input_grad,
		CudaTensor& kernel_grad, const CudaTensor& kernel, const Index3d& paddings,
		const Index3d& strides, const Real kernel_grad_scale) const;

	template CudaTensor CudaTensor::convolution_gradient<true>(const RealMemHandleConst& conv_res_grad, CudaTensor& input_grad,
		const CudaTensor& kernel, const Index3d& paddings, const Index3d& strides) const;
	template CudaTensor CudaTensor::convolution_gradient<false>(const RealMemHandleConst& conv_res_grad, CudaTensor& input_grad,
		const CudaTensor& kernel, const Index3d& paddings, const Index3d& strides) const;

	CudaTensor CudaTensor::pool(const PoolOperator& pool_operator, const Index3d& paddings,	const Index3d& strides) const
	{
		throw std::exception("Not implemented");
	}

	Index3d CudaTensor::pool(RealMemHandle result_handle, const PoolOperator& pool_operator, const Index3d& paddings,
		const Index3d& strides) const
	{
		throw std::exception("Not implemented");
	}

	CudaTensor CudaTensor::pool_input_gradient(const CudaTensor& pool_res_grad, const PoolOperator& pool_operator, const Index3d& paddings,
		const Index3d& strides) const
	{
		throw std::exception("Not implemented");
	}

	CudaTensor CudaTensor::pool_input_gradient(const RealMemHandleConst& pool_res_grad, const PoolOperator& pool_operator, const Index3d& paddings,
		const Index3d& strides) const
	{
		throw std::exception("Not implemented");
	}

	/// <summary>
	/// CUDA kernel to perform "scale-pool" operation
	/// </summary>
	/// <param name="tensor">Pointer to the input tensor (to pool from)</param>
	/// <param name="tensor_size">3d size of the input tensor</param>
	/// <param name="window_size">Pool window 3d size</param>
	/// <param name="scale_factor">Scale factor to apply to the pooled sum</param>
	/// <param name="paddings">3d size of zero paddings to be applied to the input tensor</param>
	/// <param name="result">Pointer to the array to store the result of the operation</param>
	/// <param name="result_size">3d size of the result tensor</param>
	__global__ void scale_pool_kernel(const Real* __restrict__ tensor, const Index3d tensor_size,
		const Index3d window_size, const Real scale_factor, const Index3d paddings, Real* __restrict__ result, const Index3d result_size)
	{
		const auto res_flattened_id = threadIdx.x + blockIdx.x * blockDim.x;
		const auto res_flattened_size = result_size.coord_prod();

		if (res_flattened_id >= res_flattened_size)
			return;

		const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_flattened_id, result_size);
		const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
			ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, window_size, paddings, window_size);

		auto poolled_val = Real(0);
		KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
			poolled_val += tensor[coords_to_data_id(t_x, t_y, t_z, tensor_size.y, tensor_size.z)];);

		result[res_flattened_id] = poolled_val * scale_factor;
	}

	CudaTensor CudaTensor::scale_pool(const Index3d& window_size, const Real& scale_factor) const
	{
		CudaTensor result;
		scale_pool(window_size, scale_factor, result);
		return result;
	}

	void CudaTensor::scale_pool(const Index3d& window_size, const Real& scale_factor, CudaTensor& result) const
	{
		const auto paddings = Index3d{ 0 };
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, window_size, paddings, window_size);

		result.resize(result_size);
		const auto blocks_cnt = CudaSetup::calc_blocks(result.size());
		
		scale_pool_kernel<<<blocks_cnt, CudaSetup::max_threads_per_block(), 0, cudaStreamPerThread>>>
			(begin(), tensor_size, window_size, scale_factor, paddings, result.begin(), result_size);
		CUDA_SANITY_CHECK
	}

	CudaTensor CudaTensor::average_pool(const Index3d& window_size) const
	{
		return scale_pool(window_size, static_cast<Real>(1) / window_size.coord_prod());
	}

	void CudaTensor::average_pool(const Index3d& window_size, CudaTensor& result) const
	{
		scale_pool(window_size, static_cast<Real>(1) / window_size.coord_prod(), result);
	}

	/// <summary>
	/// CUDA kernel to calculate gradient of the "scale-pool" operation with respect to its input tensor
	/// </summary>
	/// <param name="pool_res_gradient">Gradient with respect to the "scale-pool" output tensor</param>
	/// <param name="pool_res_gradient_size">3d size of the "scale-pool" output tensor (and thus, its gradient)</param>
	/// <param name="window_size">3d size of the window of the "scale-pool" operation</param>
	/// <param name="scale_factor">Scale factor of the "scale-pool' operation</param>
	/// <param name="paddings">Zero paddings that where applied to the input tensor when doing the "scale-pool" operation</param>
	/// <param name="result">Pointer to the array to store the gradient with respect to the "scale-pool" input tensor</param>
	/// <param name="result_size">3d size of the result tensor (coincide with the result of input tensor)</param>
	__global__ void scale_pool_input_gradient_kernel(const Real* __restrict__ pool_res_gradient, const Index3d pool_res_gradient_size,
		const Index3d window_size, const Real scale_factor, const Index3d paddings, Real* __restrict__ result, const Index3d result_size)
	{
		const auto res_gradient_flattened_id = threadIdx.x + blockIdx.x * blockDim.x;
		const auto res_gradient_flattened_size = pool_res_gradient_size.coord_prod();

		if (res_gradient_flattened_id >= res_gradient_flattened_size)
			return;

		const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_gradient_flattened_id, pool_res_gradient_size);
		const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
			ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, result_size, window_size, paddings, window_size);

		const auto value = pool_res_gradient[res_gradient_flattened_id] * scale_factor;
		KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
			//We assume that "pool windows" do not intersect (strides == wingow_size) and thus
			//use direct assignment ("=") instead of accumulation ("+=") in the line below
			//We also do not use atomic operations here because of the same reason
			result[coords_to_data_id(t_x, t_y, t_z, result_size.y, result_size.z)] = value;);
	}

	void CudaTensor::scale_pool_input_gradient(const CudaTensor& pool_res_gradient, const Index3d& window_size, const Real& scale_factor, CudaTensor& result) const
	{
		const auto paddings = Index3d{ 0 };
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, window_size, paddings, window_size);

		if (result_size != pool_res_gradient.size_3d())
			throw std::exception("Unexpected size of the gradient tensor");

		result.resize(tensor_size);
		result.fill_zero();

		const auto blocks_cnt = CudaSetup::calc_blocks(pool_res_gradient.size());

		scale_pool_input_gradient_kernel << <blocks_cnt, CudaSetup::max_threads_per_block(), 0, cudaStreamPerThread>> >
			(pool_res_gradient.begin(), result_size, window_size, scale_factor, paddings, result.begin(), tensor_size);
		CUDA_SANITY_CHECK
	}

	void CudaTensor::average_pool_input_gradient(const CudaTensor& pool_res_gradient, const Index3d& window_size, CudaTensor& result) const
	{
		scale_pool_input_gradient(pool_res_gradient, window_size, static_cast<Real>(1) / window_size.coord_prod(), result);
	}

	CudaTensor CudaTensor::average_pool_input_gradient(const CudaTensor& pool_res_gradient, const Index3d& window_size) const
	{
		CudaTensor result;
		scale_pool_input_gradient(pool_res_gradient, window_size, static_cast<Real>(1) / window_size.coord_prod(), result);
		return result;
	}

	template <bool MAX, bool EVAL_MAP>
	__global__ void min_max_pull_kernel(const Real* __restrict__ tensor, const Index3d tensor_size,
		const Index3d window_size, Real* __restrict__ result, const Index3d result_size, int* out_to_in_map)
	{
		const auto res_flatten_id = threadIdx.x + blockIdx.x * blockDim.x;
		const auto res_flatten_size = result_size.coord_prod();

		if (res_flatten_size <= res_flatten_id)
			return;

		const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_flatten_id, result_size);
		const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
			ConvolutionUtils::calc_kernel_loop_offsets(result_offsets,
				tensor_size, window_size, {0,0,0}, window_size);

		auto poolled_val = MAX ? -std::numeric_limits<Real>::max() : std::numeric_limits<Real>::max();
		auto poolled_id = -1;
		KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
			const int tensor_data_id = coords_to_data_id(t_x, t_y, t_z, tensor_size.y, tensor_size.z);
			const auto& current_val = tensor[tensor_data_id];
			if (MAX && (poolled_val < current_val) || !MAX && (poolled_val > current_val))
			{
				poolled_val = current_val;
				poolled_id = tensor_data_id;
			})

			if (EVAL_MAP)
				out_to_in_map[res_flatten_id] = poolled_id;

			result[res_flatten_id] = poolled_val;
	}

	std::tuple<CudaTensor, CudaArray<int>> CudaTensor::min_max_pool(const Index3d& window_size, const bool max) const
	{
		CudaTensor result;
		CudaArray<int> index_map;
		min_max_pool<true>(window_size, max, result, index_map);
		return std::make_tuple(std::move(result), std::move(index_map));
	}

	void CudaTensor::min_max_pool(const Index3d& window_size, const bool max, CudaTensor& result) const
	{
		CudaArray<int> index_map;
		min_max_pool<false>(window_size, max, result, index_map);
	}

	template <bool EVAL_MAP>
	void CudaTensor::min_max_pool(const Index3d& window_size, const bool max, CudaTensor& result, CudaArray<int>& index_map) const
	{
		const auto paddings = Index3d{ 0 };
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, window_size, paddings, window_size);

		result.resize(result_size);

		if (EVAL_MAP)
			index_map.resize(result.size());

		const auto blocks_cnt = CudaSetup::calc_blocks(result.size());

		if (max)
			min_max_pull_kernel<true, EVAL_MAP> << <blocks_cnt, CudaSetup::max_threads_per_block(), 0, cudaStreamPerThread>> >
			(begin(), tensor_size, window_size, result.begin(), result_size, index_map.begin());
		else
			min_max_pull_kernel<false, EVAL_MAP> << <blocks_cnt, CudaSetup::max_threads_per_block(), 0, cudaStreamPerThread>>>
			(begin(), tensor_size, window_size, result.begin(), result_size, index_map.begin());

		CUDA_SANITY_CHECK
	}

	template void CudaTensor::min_max_pool<true>(const Index3d& window_size, const bool max, CudaTensor& result, CudaArray<int>& index_map) const;
	template void CudaTensor::min_max_pool<false>(const Index3d& window_size, const bool max, CudaTensor& result, CudaArray<int>& index_map) const;

	void CudaTensor::min_max_pool_input_gradient(const CudaTensor& pool_res_gradient, const CudaArray<int>& out_to_in_mapping, CudaTensor& result) const
	{
		if (pool_res_gradient.size() != out_to_in_mapping.size())
			throw std::exception("Inconsistent input");

		result.resize(size_3d());
		result.fill_zero();

		thrust::scatter(thrust::cuda::par.on(cudaStreamPerThread), pool_res_gradient.begin(), pool_res_gradient.end(), out_to_in_mapping.begin(), result.begin());
		CUDA_SANITY_CHECK
	}

	CudaTensor CudaTensor::min_max_pool_input_gradient(const CudaTensor& pool_res_gradient, const CudaArray<int>& out_to_in_mapping) const
	{
		CudaTensor result;
		min_max_pool_input_gradient(pool_res_gradient, out_to_in_mapping, result);
		return result;
	}

	std::size_t CudaTensor::coords_to_data_id(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const
	{
		return _col_dim * (layer_id * _row_dim + row_id) + col_id;
	}

	bool CudaTensor::check_bounds(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const
	{
		return layer_id < _layer_dim&& row_id < _row_dim&& col_id < _col_dim;
	}

	RealMemHandleConst CudaTensor::get_layer_handle(const std::size_t& layer_id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, 0, 0))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return RealMemHandleConst(begin() + coords_to_data_id(layer_id, 0, 0), _row_dim * _col_dim);
	}

	RealMemHandle CudaTensor::get_layer_handle(const std::size_t& layer_id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, 0, 0))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return RealMemHandle(begin() + coords_to_data_id(layer_id, 0, 0), _row_dim * _col_dim);
	}

	void CudaTensor::log_layer(const std::size_t& layer_id, const std::filesystem::path& filename) const
	{
		to_host().log_layer(layer_id, filename);
	}

	void CudaTensor::log(const std::filesystem::path& directory, const std::filesystem::path& base_log_name) const
	{
		to_host().log(directory, base_log_name);
	}
}