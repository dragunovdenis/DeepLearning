//Copyright (c) 2024 Denys Dragunov, dragunovdenis@gmail.com
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

#include "ActivationFunctionHelperCuda.cuh"
#include "BasicCudaCollection.cuh"
#include <nvfunctional>
#include "thrust/transform.h"
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include "ActivationFunctionFactory.h"

namespace DeepLearning
{
	void ActivationFunctionHelperCuda::evaluate_in_place(BasicCudaCollection& collection, const ActivationFunctionId id)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread),
			collection.begin(), collection.end(), collection.begin(),
			[id] __device__ (const auto& x) {
			return ActivationFunctionFactory::make<nvstd::function<Real(Real)>>(id)(x);
		});
	}

	void ActivationFunctionHelperCuda::evaluate_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv, const ActivationFunctionId id)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), collection_func.begin(), collection_func.end(),
			thrust::make_zip_iterator(thrust::make_tuple(collection_func.begin(), collection_deriv.begin())),
			[id] __device__(const auto & x) {
			const auto res = ActivationFunctionFactory::make<nvstd::function<dual<Real>(dual<Real>)>>(id)({ x , static_cast<Real>(1)});
			return thrust::make_tuple(res.Real(), res.Dual()[0]);
		});
	}

	void ActivationFunctionHelperCuda::normalize_and_evaluate_exponent_in_place(BasicCudaCollection& collection)
	{
		const auto max_val = collection.max_element();
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread),
			collection.begin(), collection.end(), collection.begin(),
			[max_val] __device__ (const auto& x) { return std::exp(x - max_val); });
	}

	void ActivationFunctionHelperCuda::evaluate_softmax_input_grad(const BasicCudaCollection& input_exp, const BasicCudaCollection& out_grad, BasicCudaCollection& result)
	{
		const auto one_over_denominator = static_cast<Real>(1) / input_exp.sum();

		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread),
			input_exp.begin(), input_exp.end(), out_grad.begin(), result.begin(),
			[one_over_denominator] __device__ (const auto& x, const auto& y) { return x * y * one_over_denominator; });
		const auto temp_sum = result.sum() * one_over_denominator;
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread),
			result.begin(), result.end(), input_exp.begin(), result.begin(),
			[temp_sum] __device__ (const auto& x, const auto& a) { return x - a * temp_sum; });
	}

	void ActivationFunctionHelperCuda::relu_in_place(BasicCudaCollection& collection)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread),
			collection.begin(), collection.end(), collection.begin(),
			[] __device__(const auto & x) {
			return x > 0 ? x : 0;
		});
	}

	void ActivationFunctionHelperCuda::relu_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), collection_func.begin(), collection_func.end(),
			thrust::make_zip_iterator(thrust::make_tuple(collection_func.begin(), collection_deriv.begin())),
			[] __device__ (const auto & x) {
			
			return x > 0 ? thrust::make_tuple(x, 1) : thrust::make_tuple(0, 0);
		});
	}

	void ActivationFunctionHelperCuda::sigmoid_in_place(BasicCudaCollection& collection)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread),
			collection.begin(), collection.end(), collection.begin(),
			[] __device__(const auto & x) {
			return 1/(1 + std::exp(-x));
		});
	}

	void ActivationFunctionHelperCuda::sigmoid_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), collection_func.begin(), collection_func.end(),
			thrust::make_zip_iterator(thrust::make_tuple(collection_func.begin(), collection_deriv.begin())),
			[] __device__(const auto & x) {

			const auto value = 1/(1 + std::exp(-x));
			return thrust::make_tuple(value, value * (1 - value));
		});
	}

	void ActivationFunctionHelperCuda::tanh_in_place(BasicCudaCollection& collection)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread),
			collection.begin(), collection.end(), collection.begin(),
			[] __device__(const auto & x) {
			return std::tanh(x);
		});
	}

	void ActivationFunctionHelperCuda::tanh_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), collection_func.begin(), collection_func.end(),
			thrust::make_zip_iterator(thrust::make_tuple(collection_func.begin(), collection_deriv.begin())),
			[] __device__(const auto & x) {

			const auto value = std::tanh(x);
			return thrust::make_tuple(value, 1 - value * value);
		});
	}
}
