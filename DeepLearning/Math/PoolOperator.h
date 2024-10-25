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

#pragma once
#include "LinAlg3d.h"
#include "Tensor.h"
#include <msgpack.hpp>

namespace DeepLearning
{
	/// <summary>
	/// Enumerable describes different types of "pooling" operators
	/// </summary>
	enum class PoolTypeId : unsigned int {
		UNKNOWN = 0,
		MAX = 1,    //"max-pooling" operator
		MIN = 2,    //"min-pooling" operator
		AVERAGE = 3,//"average-pooling" operator
	};

	/// <summary>
	/// Returns string representation of the given pool type identifier
	/// </summary>
	std::string to_string(const PoolTypeId& pool_type_id);

	/// <summary>
	/// Parses given string to the pool type identifier
	/// </summary>
	PoolTypeId parse_pool_type(const std::string& str);


	/// <summary>
	/// Abstract interface of a "pool operator" which is supposed to facilitate such operations as 
	/// min/max/average pooling
	/// </summary>
	class PoolOperator
	{
		const Index3d _size{};
	public:
		/// <summary>
		/// Returns size of a 3d window within which the operator 
		/// </summary>
		Index3d size_3d() const;

		/// <summary>
		/// Constructor
		/// </summary>
		PoolOperator(const Index3d& size) : _size(size) { }

		/// <summary>
		/// A method to add an item
		/// </summary>
		/// <param name="id">Unique identifier of an item.
		/// It is supposed that each item can be added exactly once.</param>
		/// <param name="value">Value of the item to be taken into account by the agent</param>
		virtual void add(const Index3d& id, const Real value) = 0;

		/// <summary>
		/// Returns "pooled" value. Should be called after all the values are added to the agent
		/// </summary>
		virtual Real pool() const = 0;

		/// <summary>
		/// Resets state of the agent (so that it forgets all the values added to it)
		/// </summary>
		virtual void reset() = 0;

		/// <summary>
		/// Returns derivative of the pooled value with respect to the item with the given identifier
		/// </summary>
		/// <param name="id">Unique identifier of an item which was added to the agent by the corresponding method</param>
		virtual Real pool_deriv(const Index3d& id) const = 0;

		/// <summary>
		/// Returns clone of the current instance of an object
		/// </summary>
		virtual std::shared_ptr<PoolOperator> clone() const = 0;

		/// <summary>
		/// Factory method to instantiate pool operator by the type identifier
		/// </summary>
		/// <param name="operation_window_size">Operation window size of the pool operator to be constructed</param>
		/// <param name="pool_type_id">Pool operator type identifier</param>
		static std::unique_ptr<PoolOperator> make(const Index3d& operation_window_size, const PoolTypeId& pool_type_id);

		/// <summary>
		/// Virtual destructor to ensure that instances of descending classes do not have issues with not released resources
		/// </summary>
		virtual ~PoolOperator() {};
	};

	/// <summary>
	/// An agent to facilitate "average pool" operation
	/// </summary>
	class AveragePool : public PoolOperator
	{
		std::size_t _items_count{};
		double _sum{};
	public:

		/// <summary>
		/// Constructor
		/// </summary>
		AveragePool(const Index3d& size) : PoolOperator(size) { AveragePool::reset(); }

		void add(const Index3d& id, const Real value) override;

		Real pool() const override;

		void reset() override;

		Real pool_deriv(const Index3d& id) const override;

		std::shared_ptr<PoolOperator> clone() const override;
	};

	/// <summary>
	/// An agent to facilitate "max pool" operation
	/// </summary>
	class MaxPool : public PoolOperator
	{
	protected:
		Real _max_val{};
		Index3d _max_val_id{};
	public:
		/// <summary>
		/// Constructor
		/// </summary>
		MaxPool(const Index3d& size) : PoolOperator(size) { MaxPool::reset(); }

		void add(const Index3d& id, const Real value) override;

		Real pool() const override;

		void reset() override;

		Real pool_deriv(const Index3d& id) const override;

		std::shared_ptr<PoolOperator> clone() const override;
	};

	/// <summary>
	/// An agent to facilitate "min pool" operation
	/// </summary>
	class MinPool : public MaxPool
	{
	public:
		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="size"></param>
		MinPool(const Index3d& size) : MaxPool(size) { MinPool::reset(); }

		void add(const Index3d& id, const Real value) override;

		void reset() override;

		std::shared_ptr<PoolOperator> clone() const override;
	};


	class Tensor;

	/// <summary>
	/// Version of the pool agent that is equivalent to the convolution kernel of the corresponding size
	/// </summary>
	class KernelPool : public PoolOperator
	{
		const Tensor _kernel{};
		double _conv_result{};
	public:
		/// <summary>
		/// Constructor
		/// </summary>
		KernelPool(const Tensor& kernel) : PoolOperator(kernel.size_3d()), _kernel(kernel) 
		{
			KernelPool::reset();
		}

		void add(const Index3d& id, const Real value) override;

		Real pool() const override;

		void reset() override;

		Real pool_deriv(const Index3d& id) const override;

		std::shared_ptr<PoolOperator> clone() const override;
	};
}

MSGPACK_ADD_ENUM(DeepLearning::PoolTypeId)

