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
#include "DataContext.h"
#include "LayerData.h"
#include "LazyVector.h"
#include "IMLayerExchangeData.h"

namespace DeepLearning
{
	/// <summary>
	/// Facade to work with "trace" data of m-layers.
	/// </summary>
	template <class D = CpuDC>
	struct IMLayerTraceData
	{
		/// <summary>
		/// Virtual destructor.
		/// </summary>
		virtual ~IMLayerTraceData() = default;

		/// <summary>
		/// Number of items in the basic collection.
		/// </summary>
		virtual std::size_t size() const = 0;

		/// <summary>
		/// Access to the data of a layer with the given index.
		/// </summary>
		virtual LayerTraceData<D>& trace(const int layer_id) = 0;
	};

	/// <summary>
	/// Data container to facilitate data management of a multi-layer neural net.
	/// </summary>
	template <class D = CpuDC>
	class MLayerData : public IMLayerExchangeData<typename D::tensor_t>, public IMLayerTraceData<D>
	{
	public:

		/// <summary>
		/// The data itself.
		/// </summary>
		LazyVector<LayerData<D>> Data{};

		/// <summary>
		/// See the summary in the base class.
		/// </summary>
		LayerTraceData<D>& trace(const int layer_id) override
		{
			return Data[layer_id].Trace;
		}

		/// <summary>
		/// Number of items in the basic collection.
		/// </summary>
		std::size_t size() const override
		{
			return Data.size();
		}

		/// <summary>
		/// Resizes the underlying collection of items if needed to match the "new size".
		/// </summary>
		void resize(const std::size_t new_size) override
		{
			Data.resize(new_size);
		}

		/// <summary>
		/// Subscript operator.
		/// </summary>
		typename D::tensor_t& operator[](const std::size_t layer_id) override
		{
			return Data[layer_id].Input;
		}

		/// <summary>
		/// Subscript operator (const version).
		/// </summary>
		const typename D::tensor_t& operator[](const std::size_t layer_id) const override
		{
			return Data[layer_id].Input;
		}

		/// <summary>
		/// Assigns input data collections from the given <paramref name="source"/>.
		/// It is a responsibility of the caller to ensure that the sizes of the
		/// destination and source collections are equal.
		/// </summary>
		void assign_input(const IMLayerExchangeData<typename D::tensor_t>& source)
		{
			Data.resize(source.size());

			for (auto item_id = 0ull; item_id < Data.size(); ++item_id)
				Data[item_id].Input = source[item_id];
		}

		/// <summary>
		/// Default constructor.
		/// </summary>
		MLayerData() = default;

		/// <summary>
		/// Constructor.
		/// </summary>
		MLayerData(const std::size_t layer_count) : Data(layer_count) {}
	};
}
