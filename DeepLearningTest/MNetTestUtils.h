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

#pragma once
#include "Math/CostFunction.h"
#include "NeuralNet/MLayerData.h"

/// <summary>
/// Utility methods to test multi-net related functionality.
/// </summary>
namespace DeepLearningTest::MNetTestUtils
{
	/// <summary>
	/// Returns an instance of "input data" initialized according to the given set of parameters.
	/// </summary>
	template <class D>
	static DeepLearning::MLayerData<D> construct_random_data(const int size, const DeepLearning::Index3d& item_size)
	{
		DeepLearning::MLayerData<D> result(size);

		for (auto itemId = 0; itemId < size; ++itemId)
		{
			result[itemId].resize(item_size);
			result[itemId].init(DeepLearning::InitializationStrategy::FillRandomUniform);
		}

		return result;
	}

	/// <summary>
	/// Returns a collection of the corresponding size filled with the given <paramref name="value"/>.
	/// </summary>
	template <class D>
	static DeepLearning::LazyVector<typename D::tensor_t> construct_and_fill_vector(const long long size,
		const DeepLearning::Index3d& item_size, const DeepLearning::Real value)
	{
		DeepLearning::LazyVector<typename D::tensor_t> result(size);

		for (auto itemId = 0; itemId < size; ++itemId)
		{
			result[itemId].resize(item_size);
			result[itemId].fill(value);
		}

		return result;
	}

	/// <summary>
	/// Returns a collection of the corresponding size filled with random values from [0, 1].
	/// </summary>
	template <class D>
	static DeepLearning::LazyVector<typename D::tensor_t> construct_random_vector(const long long size, const DeepLearning::Index3d& item_size)
	{
		DeepLearning::LazyVector<typename D::tensor_t> result(size);

		for (auto itemId = 0; itemId < size; ++itemId)
		{
			result[itemId].resize(item_size);
			result[itemId].init(DeepLearning::InitializationStrategy::FillRandomUniform);
		}

		return result;
	}

	/// <summary>
	/// Returns value of cost function <paramref name="cost_func"/> evaluated
	/// on the collections <paramref name="out"/> and <paramref name="ref"/>
	/// </summary>
	template <class D>
	DeepLearning::Real evaluate_cost(const DeepLearning::CostFunction<typename D::tensor_t>& cost_func,
		const DeepLearning::LazyVector<typename D::tensor_t>& out, const DeepLearning::LazyVector<typename D::tensor_t>& ref)
	{
		if (out.size() != ref.size())
			throw std::exception("Inconsistent input.");

		DeepLearning::Real result{};

		for (auto item_id = 0ull; item_id < out.size(); ++item_id)
			result += cost_func(out[item_id], ref[item_id]);

		return result;
	}
}