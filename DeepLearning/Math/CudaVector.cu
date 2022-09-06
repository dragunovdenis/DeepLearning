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
#include <cuda_runtime.h>
#include "CudaUtils.cuh"
#include <thrust/execution_policy.h>
#include <thrust/equal.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include "CudaArray.cuh"
#include <thrust/scatter.h>
#include <thrust/iterator/constant_iterator.h>

namespace DeepLearning
{
	/// <summary>
	/// Frees the allocated memory
	/// </summary>
	void CudaVector::free()
	{
		if (_data != nullptr)
		{
			CudaUtils::cuda_free(_data);
			_data = nullptr;
		}

		_dim = 0;
		_capacity = 0;
	}

	void CudaVector::resize(const std::size_t& new_size)
	{
		if (_capacity < new_size)
		{
			free();
			_data = CudaUtils::cuda_allocate<Real>(new_size);
			_capacity = new_size;
		}

		_dim = new_size;
	}

	void CudaVector::resize(const Index3d& size_3d)
	{
		if (size_3d.x != 1ll || size_3d.y != 1ll)
			throw std::exception("Invalid input data");

		resize(size_3d.z);
	}


	void CudaVector::assign(const BasicCudaCollection& source)
	{
		resize(source.size());
		CudaUtils::cuda_copy_device2device(begin(), source.begin(), _dim);
	}

	void CudaVector::assign(const BasicCollection& source)
	{
		resize(source.size());
		CudaUtils::cuda_copy_host2device(begin(), source.begin(), _dim);
	}

	std::size_t CudaVector::size() const
	{
		return _dim;
	}

	std::size_t CudaVector::capacity() const
	{
		return _capacity;
	}

	Index3d CudaVector::size_3d() const
	{
		return { 1ull, 1ull, _dim };
	}

	Vector CudaVector::to_host() const
	{
		Vector result(size(), false/*assign zero*/);
		CudaUtils::cuda_copy_device2host(result.begin(), begin(), size());

		return result;
	}

	void CudaVector::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		Vector proxy;
		msgpack::type::make_define_array(proxy).msgpack_unpack(msgpack_o);
		assign(proxy);
	}

	CudaVector::CudaVector(const std::size_t dim, const bool assign_zero)
	{
		resize(dim);

		if (assign_zero)
			CudaUtils::fill_zero(_data, _dim);
	}

	CudaVector::CudaVector(const Index3d& size, const bool assign_zero) : 
		CudaVector(size.z, assign_zero)
	{
		if (size.x != 1ll || size.y != 1ll)
			throw std::exception("Invalid input size");
	}

	CudaVector::CudaVector(const CudaVector& vec) :
		CudaVector(vec.size(), false /*fill zero*/)
	{
		assign(vec);
	}

	CudaVector::CudaVector(const std::size_t dim, const Real range_begin, const Real range_end) :
		CudaVector(dim, false /*assign zero*/)
	{
		uniform_random_fill(range_begin, range_end);
	}

	CudaVector::CudaVector(CudaVector&& vec) noexcept : _dim(vec._dim), _capacity(vec._capacity)
	{
		_data = vec._data;
		vec.abandon_resources();
	}

	CudaVector& CudaVector::operator=(const CudaVector& vec)
	{
		if (this != &vec)
			assign(vec);

		return *this;
	}

	CudaVector& CudaVector::operator=(CudaVector&& vec) noexcept
	{
		if (this != &vec)
		{
			free();
			_dim = vec._dim;
			_capacity = vec._capacity;
			_data = vec._data;
			vec.abandon_resources();
		}

		return *this;
	}

	CudaVector::~CudaVector()
	{
		free();
	}

	std::size_t CudaVector::dim() const
	{
		return _dim;
	}

	void CudaVector::abandon_resources()
	{
		_data = nullptr;
		free();
	}

	CudaVector& CudaVector::operator += (const CudaVector& vec)
	{
		add(vec);
		return *this;
	}

	CudaVector& CudaVector::operator -= (const CudaVector& vec)
	{
		sub(vec);
		return *this;
	}

	CudaVector& CudaVector::operator *= (const Real& scalar)
	{
		mul(scalar);
		return *this;
	}

	bool CudaVector::operator == (const CudaVector & vect) const
	{
		return size() == vect.size() &&
			   thrust::equal(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), vect.begin());
	}

	bool CudaVector::operator !=(const CudaVector& vect) const
	{
		return !(*this == vect);
	}

	CudaVector CudaVector::random(const std::size_t dim, const Real range_begin, const Real range_end)
	{
		return CudaVector(dim, range_begin, range_end);
	}

	void CudaVector::log(const std::filesystem::path& file_name) const
	{
		to_host().log(file_name);
	}

	CudaVector operator + (const CudaVector& vec1, const CudaVector& vec2)
	{
		auto result = vec1;
		return result += vec2;
	}

	CudaVector operator -(const CudaVector& vec1, const CudaVector& vec2)
	{
		auto result = vec1;
		return result -= vec2;
	}

	CudaVector operator *(const CudaVector& vec, const Real& scalar)
	{
		auto result = vec;
		return result *= scalar;
	}

	CudaVector operator *(const Real& scalar, const CudaVector& vec)
	{
		return vec * scalar;
	}

	void CudaVector::generate_with_random_selection_map(const std::size_t& selected_cnt, CudaArray<int>& aux_collection)
	{
		if (selected_cnt >= size())
		{
			fill(Real(1));
			return;
		}

		aux_collection.resize(size());
		thrust::sequence(thrust::cuda::par.on(cudaStreamPerThread), aux_collection.begin(), aux_collection.end(), 0);
		uniform_random_fill(Real(-1), Real(1)); //fill the current collection with random values
		//and use it as a key collection in the following sorting procedure
		thrust::sort_by_key(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), aux_collection.begin());

		const auto one_iterator = thrust::make_constant_iterator(Real(1));
		thrust::scatter(thrust::cuda::par.on(cudaStreamPerThread), one_iterator, one_iterator + static_cast<int>(selected_cnt),
			aux_collection.begin(), begin());

		const auto zero_iterator = thrust::make_constant_iterator(Real(0));
		thrust::scatter(thrust::cuda::par.on(cudaStreamPerThread), zero_iterator, zero_iterator + static_cast<int>(size() - selected_cnt),
			aux_collection.begin() + static_cast<int>(selected_cnt), begin());
	}

	void CudaVector::generate_with_random_selection_map(const std::size_t& selected_cnt)
	{
		CudaArray<int> aux_collection;
		generate_with_random_selection_map(selected_cnt, aux_collection);
	}

}