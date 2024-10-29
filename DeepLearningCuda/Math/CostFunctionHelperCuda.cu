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

#include "CostFunctionHelperCuda.cuh"
#include "BasicCudaCollection.cuh"
#include "thrust/reduce.h"
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <nvfunctional>
#include "CudaVector.cuh"
#include <Math/CostFunctionFactory.h>

namespace DeepLearning
{
	Real CostFunctionHelperCuda::evaluate_cost(const BasicCudaCollection& output, const BasicCudaCollection& reference, const CostFunctionId id)
	{
		thread_local CudaVector temp;
		temp.resize(output.size());

		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), output.begin(), output.end(), reference.begin(), temp.begin(),
			[id] __device__(const auto&x, const auto& y) {
			const auto func = CostFunctionFactory::make<nvstd::function<Real(Real, Real)>>(id);
			return func(x, y);
		});

		return thrust::reduce(thrust::cuda::par.on(cudaStreamPerThread), temp.begin(), temp.end(), static_cast<Real>(0), thrust::plus<Real>());
	}

	Real CostFunctionHelperCuda::evaluate_cost_and_gradient(BasicCudaCollection& output, const BasicCudaCollection& reference, const CostFunctionId id)
	{
		//TODO: think about more efficient solution (although this method is not supposed to be used in the training of a neural network)
		const auto func_val = evaluate_cost(output, reference, id);
		evaluate_gradient(output, reference, id);

		return func_val;
	}

	void CostFunctionHelperCuda::evaluate_gradient(BasicCudaCollection& output, const BasicCudaCollection& reference, const CostFunctionId id)
	{
		thrust::transform(thrust::cuda::par.on(cudaStreamPerThread), output.begin(), output.end(), reference.begin(), output.begin(),
			[id] __device__(const auto & x, const auto & ref) {
			const auto func = CostFunctionFactory::make<nvstd::function<dual<Real>(dual<Real>, Real)>>(id);
			return  func({ x, static_cast<Real>(1) }, ref).Dual()[0];
		});
	}
}