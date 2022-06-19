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
#include "CudaUtils.cuh"
#include <thrust/execution_policy.h>
#include <thrust/equal.h>

namespace DeepLearning
{
	void CudaTensor::free()
	{
		if (_data != nullptr)
		{
			gpuErrchk(cudaFree(_data));
			_data = nullptr;
		}

		_row_dim = 0;
		_col_dim = 0;
		_layer_dim = 0;
	}

	void CudaTensor::abandon_resources()
	{
		_data = nullptr;
		free();
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

	void CudaTensor::resize(const std::size_t& new_layer_dim, const std::size_t& new_row_dim, const std::size_t& new_col_dim)
	{
		const auto new_size = new_layer_dim * new_row_dim * new_col_dim;
		if (size() != new_size)
		{
			free();
			_data = CudaUtils::cuda_allocate<Real>(new_size);
		}

		_layer_dim = new_layer_dim;
		_row_dim = new_row_dim;
		_col_dim = new_col_dim;
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
		const std::size_t col_dim, const bool assign_zero) :
		_layer_dim(layer_dim), _row_dim(row_dim), _col_dim(col_dim)
	{
		_data = CudaUtils::cuda_allocate<Real>(size());

		if (assign_zero)
			CudaUtils::fill_zero(_data, size());
	}

	CudaTensor::CudaTensor(const Index3d& size, const bool assign_zero) :
		CudaTensor(size.x, size.y, size.z, assign_zero)
	{}

	CudaTensor::CudaTensor(const CudaTensor& tensor) :
		CudaTensor(tensor.layer_dim(), tensor.row_dim(), tensor.col_dim(), false /*assign zero*/)
	{
		CudaUtils::cuda_copy_device2device(begin(), tensor.begin(), size());
	}

	CudaTensor::CudaTensor(CudaVector&& vector) noexcept :
		_layer_dim(1), _row_dim(1), _col_dim(vector.dim())
	{
		_data = vector.begin();
		vector.abandon_resources();
	}

	CudaTensor::CudaTensor(CudaMatrix&& matrix) noexcept :
		_layer_dim(1), _row_dim(matrix.row_dim()), _col_dim(matrix.col_dim())
	{
		_data = matrix.begin();
		matrix.abandon_resources();
	}

	CudaTensor::CudaTensor(CudaTensor&& tensor) noexcept :
		_layer_dim(tensor.layer_dim()), _row_dim(tensor.row_dim()), _col_dim(tensor.col_dim())
	{
		_data = tensor.begin();
		tensor.abandon_resources();
	}

	CudaTensor::CudaTensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const Real range_begin, const Real range_end) : 
		CudaTensor(layer_dim, row_dim, col_dim, false /*assign zero*/)
	{
		uniform_random_fill(range_begin, range_end);
	}

	CudaTensor::CudaTensor(const Index3d& size, const Real range_begin, const Real range_end) :
		CudaTensor(size.x, size.y, size.z, range_begin, range_end)
	{}

	CudaTensor& CudaTensor::operator =(const CudaTensor& tensor)
	{
		assign(tensor);
		return *this;
	}

	CudaTensor& CudaTensor::operator =(CudaVector&& vector) noexcept
	{
		resize(1, 1, vector.dim());
		_data = vector.begin();
		vector.abandon_resources();

		return *this;
	}

	CudaTensor& CudaTensor::operator =(CudaMatrix&& matrix) noexcept
	{
		resize(1, matrix.row_dim(), matrix.col_dim());
		_data = matrix.begin();
		matrix.abandon_resources();

		return *this;
	}

	CudaTensor& CudaTensor::operator =(CudaTensor&& tensor) noexcept
	{
		resize(tensor.layer_dim(), tensor.row_dim(), tensor.col_dim());
		_data = tensor.begin();
		tensor.abandon_resources();

		return *this;
	}

	CudaTensor::~CudaTensor()
	{
		free();
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

	std::vector<CudaTensor>& operator +=(std::vector<CudaTensor>& op1, const std::vector<CudaTensor>& op2)
	{
		if (op1.size() != op2.size())
			throw std::exception("Incompatible input");

		for (auto op_id = 0ull; op_id < op1.size(); op_id++)
			op1[op_id] += op2[op_id];

		return op1;
	}

	std::vector<CudaTensor>& operator *=(std::vector<CudaTensor>& op1, const Real& scalar)
	{
		for (auto op_id = 0ull; op_id < op1.size(); op_id++)
			op1[op_id] *= scalar;

		return op1;
	}

	bool CudaTensor::operator ==(const CudaTensor& tensor) const
	{
		return  layer_dim() == tensor.layer_dim() &&
			    row_dim() == tensor.row_dim() &&
			    col_dim() == tensor.col_dim() &&
			    thrust::equal(thrust::device, begin(), end(), tensor.begin());
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

	Index3d CudaTensor::convolve(RealMemHandle result_handle, const CudaTensor& kernel, const Index3d& paddings, const Index3d& strides) const
	{
		throw std::exception("Not implemented");
	}

	CudaTensor CudaTensor::convolve(const CudaTensor& kernel, const Index3d& paddings, const Index3d& strides) const
	{
		throw std::exception("Not implemented");
	}

	std::tuple<CudaTensor, CudaTensor> CudaTensor::convolution_gradient(const CudaTensor& conv_res_grad, const CudaTensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		throw std::exception("Not implemented");
	}

	std::tuple<CudaTensor, CudaTensor> CudaTensor::convolution_gradient(const RealMemHandleConst& conv_res_grad, const CudaTensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		throw std::exception("Not implemented");
	}

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

	std::tuple<CudaTensor, std::vector<std::size_t>> CudaTensor::min_max_pool_2d(const Index2d& window_size, const bool max) const
	{
		throw std::exception("Not implemented");
	}

	CudaTensor CudaTensor::min_max_pool_2d_input_gradient(const CudaTensor& pool_res_gradient, const std::vector<std::size_t>& out_to_in_mapping) const
	{
		throw std::exception("Not implemented");
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

		return RealMemHandleConst(_data + coords_to_data_id(layer_id, 0, 0), _row_dim * _col_dim);
	}

	RealMemHandle CudaTensor::get_layer_handle(const std::size_t& layer_id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, 0, 0))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return RealMemHandle(_data + coords_to_data_id(layer_id, 0, 0), _row_dim * _col_dim);
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