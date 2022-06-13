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

#include "CudaMatrix.cuh"
#include "CudaVector.cuh"
#include "CudaUtils.cuh"
#include "Vector.h"
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/equal.h>
#include "CudaSetup.h"

namespace DeepLearning
{
	void CudaMatrix::free()
	{
		if (_data != nullptr)
		{
			gpuErrchk(cudaFree(_data));
			_data = nullptr;
		}

		_row_dim = 0;
		_col_dim = 0;
	}

	/// <summary>
	/// Total number of elements in the matrix
	/// </summary>
	std::size_t CudaMatrix::size() const
	{
		return _row_dim * _col_dim;
	}

	Matrix CudaMatrix::to_host() const
	{
		Matrix result(_row_dim, _col_dim, false /*assign zero*/);
		CudaUtils::cuda_copy_device2host(result.begin(), begin(), size());

		return result;
	}

	void CudaMatrix::resize(const std::size_t& new_row_dim, const std::size_t& new_col_dim)
	{
		const auto _new_size = new_row_dim * new_col_dim;
		if (size() != _new_size)
		{
			free();
			_data = CudaUtils::cuda_allocate<Real>(_new_size);
		}

		_row_dim = new_row_dim;
		_col_dim = new_col_dim;

	}

	void CudaMatrix::assign(const CudaMatrix& source)
	{
		resize(source.row_dim(), source.col_dim());
		CudaUtils::cuda_copy_device2device(begin(), source.begin(), size());
	}

	void CudaMatrix::assign(const Matrix& source)
	{
		resize(source.row_dim(), source.col_dim());
		CudaUtils::cuda_copy_host2device(begin(), source.begin(), size());
	}


	void CudaMatrix::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		Matrix proxy;
		msgpack::type::make_define_array(proxy).msgpack_unpack(msgpack_o);
		assign(proxy);
	}

	std::size_t CudaMatrix::col_dim() const
	{
		return _col_dim;
	}

	std::size_t CudaMatrix::row_dim() const
	{
		return _row_dim;
	}

	CudaMatrix::CudaMatrix(const std::size_t row_dim, const std::size_t col_dim, const bool assign_zero) :
		_row_dim(row_dim), _col_dim(col_dim)
	{
		_data = CudaUtils::cuda_allocate<Real>(size());

		if (assign_zero)
			CudaUtils::fill_zero(_data, size());
	}

	CudaMatrix::CudaMatrix(const std::size_t row_dim, const std::size_t col_dim,
		const Real range_begin, const Real range_end) : CudaMatrix(row_dim, col_dim, false /*assign zero*/)
	{
		uniform_random_fill(range_begin, range_end);
	}

	CudaMatrix::CudaMatrix(const CudaMatrix& matr) :
		CudaMatrix(matr.row_dim(), matr.col_dim(), false /*assign zero*/)
	{
		CudaUtils::cuda_copy_device2device(begin(), matr.begin(), size());
	}

	void CudaMatrix::abandon_resources()
	{
		_data = nullptr;
		_col_dim = 0;
		_row_dim = 0;
	}

	CudaMatrix::CudaMatrix(CudaMatrix&& matr) noexcept :
		_row_dim(matr.row_dim()), _col_dim(matr.col_dim())
	{
		_data = matr._data;
		matr.abandon_resources();
	}

	CudaMatrix& CudaMatrix::operator =(const CudaMatrix& matr)
	{
		assign(matr);
		return *this;
	}

	CudaMatrix::~CudaMatrix()
	{
		free();
	}

	bool CudaMatrix::operator ==(const CudaMatrix& matr) const
	{
		if (_row_dim != matr.row_dim() || _col_dim != matr.col_dim())
			return false;

		return thrust::equal(thrust::device,  begin(), end(), matr.begin());
	}

	bool CudaMatrix::operator !=(const CudaMatrix& matr) const
	{
		return !(*this == matr);
	}

	CudaMatrix CudaMatrix::random(const std::size_t row_dim, const std::size_t col_dim, const Real range_begin, const Real range_end)
	{
		return CudaMatrix(row_dim, col_dim, range_begin, range_end);
	}

	CudaMatrix& CudaMatrix::operator +=(const CudaMatrix& mat)
	{
		add(mat);
		return *this;
	}

	CudaMatrix& CudaMatrix::operator -=(const CudaMatrix& mat)
	{
		sub(mat);
		return *this;
	}

	CudaMatrix& CudaMatrix::operator *=(const Real& scalar)
	{
		mul(scalar);
		return *this;
	}

	void CudaMatrix::log(const std::filesystem::path& file_name) const
	{
		to_host().log(file_name);
	}

	CudaMatrix operator +(const CudaMatrix& mat1, const CudaMatrix& mat2)
	{
		auto result = mat1;
		return result += mat2;
	}

	CudaMatrix operator -(const CudaMatrix& mat1, const CudaMatrix& mat2)
	{
		auto result = mat1;
		return result -= mat2;
	}

	CudaMatrix operator *(const CudaMatrix& mat, const Real& scalar)
	{
		auto result = mat;
		return result *= scalar;
	}

	CudaMatrix operator *(const Real& scalar, const CudaMatrix& mat)
	{
		return mat * scalar;
	}

	/// <summary>
	/// CUDA kernel to perform matrix vector multiplication
	/// </summary>
	/// <param name="row_dim">Number of rows in the input matrix</param>
	/// <param name="col_dim">Number of columns in the input matrix (must coincide with the vector size)</param>
	/// <param name="matr_data">Matrix operand</param>
	/// <param name="vect_data">Vector operand</param>
	/// <param name="result">Result of the matrix-vector multiplication</param>
	__global__  void matrix_vector_multiply_kernel(const int row_dim, const int col_dim,
		const Real* matr_data, const Real* vect_data, Real* result)
	{
		const int row_id = blockIdx.x * blockDim.x + threadIdx.x;

		if (row_id >= row_dim)
			return;

		Real accumulator = Real(0.0);

		for (int col_id = 0; col_id < col_dim; col_id++)
			accumulator += matr_data[row_id * col_dim + col_id] * vect_data[col_id];

		result[row_id] = accumulator;
	}

	CudaVector operator *(const CudaMatrix& matr, const CudaVector& vec)
	{
		if (matr.col_dim() != vec.dim())
			throw std::exception("Incompatible input data");

		CudaVector result(matr.row_dim(), false /*assign zero*/);

		const auto blocks_cnt = CudaSetup::calc_blocks(matr.row_dim());
		const auto threads_per_block = CudaSetup::threads_per_block();
		matrix_vector_multiply_kernel << <blocks_cnt, threads_per_block >> > (
			static_cast<int>(matr.row_dim()), static_cast<int>(matr.col_dim()),
			matr.begin(), vec.begin(), result.begin());

		return result;
	}

	/// <summary>
	/// CUDA kernel to perform matrix vector multiplication with "simultaneous" summation with the given vector
	/// </summary>
	/// <param name="row_dim">Number of rows in the input matrix</param>
	/// <param name="col_dim">Number of columns in the input matrix (must coincide with the vector size)</param>
	/// <param name="matr_data">Matrix operand</param>
	/// <param name="vect_to_mul_data">Vector operand to multiply</param>
	/// <param name="vect_to_add_data">Vector operand to add</param>
	/// <param name="result">Result of the matrix-vector multiplication</param>
	__global__  void matrix_vector_multiply_add_kernel(const int row_dim, const int col_dim,
		const Real* matr_data, const Real* mul_vect_data, const Real* add_vect_data, Real* result)
	{
		const int row_id = blockIdx.x * blockDim.x + threadIdx.x;

		if (row_id >= row_dim)
			return;

		Real accumulator = Real(0.0);

		for (int col_id = 0; col_id < col_dim; col_id++)
			accumulator += matr_data[row_id * col_dim + col_id] * mul_vect_data[col_id];

		result[row_id] = accumulator + add_vect_data[row_id];
	}

	CudaVector CudaMatrix::mul_add(const BasicCudaCollection& mul_vec, const BasicCudaCollection& add_vec) const
	{
		if (col_dim() != mul_vec.size() || row_dim() != add_vec.size())
			throw std::exception("Incompatible input data");


		CudaVector result(row_dim(), false /*assign zero*/);

		const auto blocks_cnt = CudaSetup::calc_blocks(row_dim());
		const auto threads_per_block = CudaSetup::threads_per_block();
		matrix_vector_multiply_add_kernel << <blocks_cnt, threads_per_block >> > (
			static_cast<int>(row_dim()), static_cast<int>(col_dim()),
			begin(), mul_vec.begin(), add_vec.begin(), result.begin());

		return result;
	}

	/// <summary>
	/// CUDA kernel to perform vector-matrix multiplication (vector from the left)
	/// </summary>
	/// <param name="row_dim">Number of rows in the input matrix</param>
	/// <param name="col_dim">Number of columns in the input matrix (must coincide with the vector size)</param>
	/// <param name="matr_data">Matrix operand</param>
	/// <param name="vect_data">Vector operand</param>
	/// <param name="result">Result of the matrix-vector multiplication</param>
	__global__  void vector_matrix_multiply_kernel(const int row_dim, const int col_dim,
		const Real* matr_data, const Real* vect_data, Real* result)
	{
		const int col_id = blockIdx.x * blockDim.x + threadIdx.x;

		if (col_id >= col_dim)
			return;

		Real accumulator = Real(0.0);

		for (int row_id = 0; row_id < row_dim; row_id++)
			accumulator += matr_data[row_id * col_dim + col_id] * vect_data[row_id];

		result[col_id] = accumulator;
	}

	CudaVector operator *(const BasicCudaCollection& vec, const CudaMatrix& matr)
	{
		if (matr.row_dim() != vec.size())
			throw std::exception("Incompatible input data");

		CudaVector result(matr.col_dim(), false /*assign zero*/);

		const auto blocks_cnt = CudaSetup::calc_blocks(matr.col_dim());
		const auto threads_per_block = CudaSetup::threads_per_block();
		vector_matrix_multiply_kernel << <blocks_cnt, threads_per_block >> > (
			static_cast<int>(matr.row_dim()), static_cast<int>(matr.col_dim()),
			matr.begin(), vec.begin(), result.begin());

		return result;
	}

	/// <summary>
	/// CUDA kernel to perform vector-column by vector-row multiplication
	/// </summary>
	/// <param name="vec_col_size">Size of the "column" vector</param>
	/// <param name="vec_col_data">Pointer to the array of elements of the "column" vector</param>
	/// <param name="vec_row_size">Size of the "row" vector</param>
	/// <param name="vec_row_data">Pointer to the array of elements of the "row" vector</param>
	/// <param name="result">Pointer to the array of "result" (i.e. vec_col_size x vec_row_size matrix)</param>
	__global__ void vector_col_times_vector_row_kernel(const int vec_col_size, const Real* vec_col_data,
		const int vec_row_size, const Real* vec_row_data, Real* result)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;

		if (id >= vec_col_size * vec_row_size) return;

		const auto row_id = id / vec_row_size;
		const auto col_id = id % vec_row_size;

		result[id] = vec_row_data[col_id] * vec_col_data[row_id];
	}

	CudaMatrix vector_col_times_vector_row(const BasicCudaCollection& vec_col, const BasicCudaCollection& vec_row)
	{
		CudaMatrix result(vec_col.size(), vec_row.size(), false /*assign zero*/);

		const auto blocks_cnt = CudaSetup::calc_blocks(result.size());
		const auto threads_per_block = CudaSetup::threads_per_block();

		vector_col_times_vector_row_kernel << <blocks_cnt, threads_per_block >> > (vec_col.size(), vec_col.begin(),
			vec_row.size(), vec_row.begin(), result.begin());

		return result;
	}
}