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

#include "BasicCudaCollection.cuh"
#include <nvfunctional>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/generate.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/logical.h>
#include "CudaUtils.cuh"
#include <algorithm>
#include <exception>
#include <random>
#include "../Utilities.h"

namespace DeepLearning
{
	void BasicCudaCollection::add(const BasicCudaCollection& collection)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), collection.begin(), begin(), thrust::plus<Real>());
		CUDA_SANITY_CHECK
	}

	void BasicCudaCollection::add_scaled(const BasicCudaCollection& collection, const Real& scalar)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), collection.begin(), begin(), [scalar] __device__ (const auto& x, const auto& y) { return x + scalar * y; });
		CUDA_SANITY_CHECK
	}

	void BasicCudaCollection::scale_and_add(const BasicCudaCollection& collection, const Real& scalar)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), collection.begin(), begin(), [scalar] __device__ (const auto& x, const auto& y) { return x * scalar +  y; });
		CUDA_SANITY_CHECK
	}

	void BasicCudaCollection::sub(const BasicCudaCollection& collection)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), collection.begin(), begin(), [] __device__ (const auto& x, const auto& y) { return x - y; });
		CUDA_SANITY_CHECK
	}

	void BasicCudaCollection::mul(const Real& scalar)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), begin(), [scalar] __device__ (const auto& x ) { return x * scalar; });
		CUDA_SANITY_CHECK
	}

	/// <summary>
	/// Host/device version of the absolute value function
	/// </summary>
	template <class R>
	__device__ __host__ R cuda_abs(const R& x)
	{
		return x >= R(0) ? x : -x;
	}

	Real BasicCudaCollection::max_abs() const
	{
		if (empty())
			return std::numeric_limits<Real>::signaling_NaN();

		return std::abs(thrust::reduce(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), Real(0),
			[] __device__ (const auto& x, const auto& y) {
				if (cuda_abs(x) > cuda_abs(y))
					return x;

				return y;
			}
		));
		CUDA_SANITY_CHECK
	}

	Real BasicCudaCollection::sum() const
	{
		const auto result = thrust::reduce(thrust::cuda::par.on(cudaStreamPerThread), begin(), end());
		CUDA_SANITY_CHECK
		return result;
	}

	Real BasicCudaCollection::sum_of_squares() const
	{
		const auto result = thrust::transform_reduce(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), []__device__(const auto & x) { return x * x; },
			Real(0), thrust::plus<Real>());
		CUDA_SANITY_CHECK
		return result;
	}

	void BasicCudaCollection::fill(const Real& val)
	{
		thrust::fill(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), val);
	}

	bool BasicCudaCollection::empty() const
	{
		return size() == 0;
	}

	Real* BasicCudaCollection::begin()
	{
		return _data;
	}

	const Real* BasicCudaCollection::begin() const
	{
		return _data;
	}

	Real* BasicCudaCollection::end()
	{
		return _data + size();
	}

	const Real* BasicCudaCollection::end() const
	{
		return _data + size();
	}

	void BasicCudaCollection::hadamard_prod_in_place(const BasicCudaCollection& collection)
	{
		if (size() != collection.size())
			throw std::exception("Inconsistent input");

		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), collection.begin(), begin(), [] __device__ (const auto& x, const auto& y) { return x * y; });
		CUDA_SANITY_CHECK
	}

	Real BasicCudaCollection::dot_product(const BasicCudaCollection& collection) const
	{
		const auto result = thrust::inner_product(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), collection.begin(), Real(0));
		CUDA_SANITY_CHECK
		return result;
	}

	std::size_t BasicCudaCollection::max_element_id() const
	{
		const auto id = thrust::max_element(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(),
			[] __device__(const auto & x, const auto & y) { return x < y; }) - begin();
		CUDA_SANITY_CHECK
		return static_cast<std::size_t>(id);
	}

	Real BasicCudaCollection::max_element() const
	{
		if (empty())
			return std::numeric_limits<Real>::quiet_NaN();

		return thrust::reduce(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), Real(0),
			[] __device__(const auto & x, const auto & y) {
			return  (x < y) ? y : x;
		});
		CUDA_SANITY_CHECK
	}

	std::vector<Real> BasicCudaCollection::to_stdvector() const
	{
		std::vector<Real> result(size());
		gpuErrchk(cudaMemcpyAsync(result.data(), begin(), size() * sizeof(Real), cudaMemcpyKind::cudaMemcpyDeviceToHost, cudaStreamPerThread));

		return result;
	}

	RealMemHandleConst BasicCudaCollection::get_handle() const
	{
		return RealMemHandleConst(begin(), size());
	}

	RealMemHandle BasicCudaCollection::get_handle()
	{
		return RealMemHandle(begin(), size());
	}

	/// <summary>
	/// Generator of normally distributed random numbers
	/// </summary>
	struct NormalRandGen
	{
		Real mean{};
		Real sigma{};
		int seed;

		/// <summary>
		/// The operator
		/// </summary>
		__device__	Real operator () (int idx)
		{
			thrust::minstd_rand rng(seed);
			thrust::random::normal_distribution<Real> dist(mean, sigma);
			rng.discard(idx);
			return dist(rng);
		}
	};

	/// <summary>
	/// Generator of uniformly distributed random numbers
	/// </summary>
	struct UniformRandGen
	{
		Real min{};
		Real max{};
		int seed{};

		/// <summary>
		/// The operator
		/// </summary>
		__device__	Real operator () (int idx)
		{
			thrust::default_random_engine rng(seed);
			thrust::uniform_real_distribution<Real> dist(min, max);
			rng.discard(idx);
			return dist(rng);
		}
	};

	void BasicCudaCollection::standard_random_fill(const Real& sigma)
	{
		if (empty())
			return;

		const auto stddev = sigma < Real(0) ? Real(1) / Real(std::sqrt(size())) : sigma;
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(static_cast<int>(size())), begin(), NormalRandGen{0, stddev, Utils::get_random_int()});
		CUDA_SANITY_CHECK
	}

	void BasicCudaCollection::uniform_random_fill(const Real& min, const Real& max)
	{
		if (empty())
			return;

		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(static_cast<int>(size())), begin(), UniformRandGen{ min, max, Utils::get_random_int()});
		CUDA_SANITY_CHECK
	}

	bool BasicCudaCollection::is_nan() const
	{
		const auto result = thrust::any_of(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), [] __device__ (const auto& x) { return isnan(x); });
		CUDA_SANITY_CHECK
		return result;
	}

	bool BasicCudaCollection::is_inf() const
	{
		const auto result = thrust::any_of(thrust::cuda::par.on(cudaStreamPerThread), begin(), end(), [] __device__ (const auto& x) { return isinf(x); });
		CUDA_SANITY_CHECK
		return result;
	}
}