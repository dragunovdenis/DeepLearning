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

#include "ActivationFunction.h"
#include "BasicCudaCollection.cuh"
#include <nvfunctional>
#include "thrust/transform.h"
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace DeepLearning::ActivationFunctionHelper
{
	void evaluate_in_place(BasicCudaCollection& collection, const ActivationFunctionId id)
	{
		thrust::transform(thrust::device, collection.begin(), collection.end(), collection.begin(),
			[id] __device__ (const auto& x) {
			return make<nvstd::function<Real(Real)>>(id)(x);
		});
	}

	void evaluate_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv, const ActivationFunctionId id)
	{
		thrust::transform(thrust::device, collection_func.begin(), collection_func.end(),
			thrust::make_zip_iterator(thrust::make_tuple(collection_func.begin(), collection_deriv.begin())),
			[id] __device__(const auto & x) {
			const auto res = make<nvstd::function<dual<Real>(dual<Real>)>>(id)({ x , Real(1)});
			return thrust::make_tuple(res.Real(), res.Dual()[0]);
		});
	}
}