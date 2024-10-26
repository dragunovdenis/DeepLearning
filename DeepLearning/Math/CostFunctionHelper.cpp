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

#include "CostFunctionHelper.h"
#include "CostFunctionFactory.h"
#include "BasicCollection.h"
#include <functional>
#include <numeric>
#include "Dual.h"

namespace DeepLearning
{
	Real CostFunctionHelper::evaluate_cost(const BasicCollection& output, const BasicCollection& reference, const CostFunctionId id)
	{
		const auto func = CostFunctionFactory::make<std::function<Real(Real, Real)>>(id);

		const auto result = std::transform_reduce(output.begin(), output.end(), reference.begin(), Real(0), std::plus<Real>(),
			[&](const auto& x, const auto& ref) { return  func(x, ref); });
		return result;
	}

	Real CostFunctionHelper::evaluate_cost_and_gradient(BasicCollection& output, const BasicCollection& reference, const CostFunctionId id)
	{
		auto func_val = Real(0);

		const auto func = CostFunctionFactory::make<std::function<dual<Real>(dual<Real>, Real)>>(id);

		for (auto item_id = 0ull; item_id < output.size(); item_id++)
		{
			const auto res = func({ output.begin()[item_id] , Real(1) }, reference.begin()[item_id]);
			func_val += res.Real();
			output.begin()[item_id] = res.Dual()[0];
		}

		return func_val;
	}

	void CostFunctionHelper::evaluate_gradient(BasicCollection& output, const BasicCollection& reference, const CostFunctionId id)
	{
		const auto func = CostFunctionFactory::make<std::function<dual<Real>(dual<Real>, Real)>>(id);
		std::transform(output.begin(), output.end(), reference.begin(), output.begin(), [&](const auto& x, const auto& ref) {
			return  func({ x, Real(1) }, ref).Dual()[0]; });
	}
}
