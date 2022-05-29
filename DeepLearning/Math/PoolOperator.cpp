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

#include "PoolOperator.h"
#include <exception>

namespace DeepLearning
{
	Index3d PoolOperator::size_3d() const
	{
		return _size;
	}

	std::unique_ptr<PoolOperator> PoolOperator::make(const Index3d& operation_window_size, const PoolTypeId& pool_type_id)
	{
		switch (pool_type_id)
		{
			case PoolTypeId::MAX:
				return std::make_unique<MaxPool>(operation_window_size);
			case PoolTypeId::AVERAGE: 
				return std::make_unique<AveragePool>(operation_window_size);
			default:
				throw std::exception("Not implemented");
		}
	}

	void AveragePool::add(const Index3d& id, const Real value)
	{
		_items_count++;
		_sum += value;
	}

	Real AveragePool::pool() const
	{
#ifdef DEBUG
		if (_items_count == 0)
			throw std::exception("No items to pool from");
#endif // DEBUG

		return _sum / _items_count;
	}

	void AveragePool::reset()
	{
		_items_count = 0ull;
		_sum = Real(0);
	}

	Real AveragePool::pool_deriv(const Index3d& id) const
	{
#ifdef DEBUG
		if (_items_count == 0)
			throw std::exception("No items to pool from");
#endif // DEBUG

		return Real(1) / _items_count;
	}

	std::shared_ptr<PoolOperator> AveragePool::clone() const
	{
		return std::make_shared<AveragePool>(*this);
	}

	void MaxPool::add(const Index3d& id, const Real value)
	{
		if (value > _max_val)
		{
			_max_val = value;
			_max_val_id = id;
		}
	}

	Real MaxPool::pool() const
	{
		return _max_val;
	}

	void MaxPool::reset()
	{
		_max_val = -std::numeric_limits<Real>::max();
		_max_val_id = { -1, -1, -1 };
	}

	Real MaxPool::pool_deriv(const Index3d& id) const
	{
		if (id == _max_val_id)
			return Real(1);

		return Real(0);
	}

	std::shared_ptr<PoolOperator> MaxPool::clone() const
	{
		return std::make_shared<MaxPool>(*this);
	}

	void MinPool::add(const Index3d& id, const Real value)
	{
		if (value < _max_val)
		{
			_max_val = value;
			_max_val_id = id;
		}
	}

	void MinPool::reset()
	{
		_max_val = std::numeric_limits<Real>::max();
		_max_val_id = { -1, -1, -1 };
	}

	std::shared_ptr<PoolOperator> MinPool::clone() const
	{
		return std::make_shared<MinPool>(*this);
	}


	void KernelPool::add(const Index3d& id, const Real value)
	{
		_conv_result += _kernel(id.x, id.y, id.z) * value;
	}

	Real KernelPool::pool() const
	{
		return _conv_result;
	}

	void KernelPool::reset()
	{
		_conv_result = Real(0);
	}

	Real KernelPool::pool_deriv(const Index3d& id) const
	{
		return _kernel(id.x, id.y, id.z);
	}

	std::shared_ptr<PoolOperator> KernelPool::clone() const
	{
		return std::make_shared<KernelPool>(*this);
	}

	std::string to_string(const PoolTypeId& pool_type_id)
	{
		switch (pool_type_id)
		{
		case PoolTypeId::MAX: return "MAX";
		case PoolTypeId::MIN: return "MIN";
		case PoolTypeId::AVERAGE: return "AVERAGE";
		default:
			return "UNKNOWN";
		}
	}

	PoolTypeId parse_pool_type(const std::string& str)
	{
		const auto str_normalized = Utils::normalize_string(str);

		for (unsigned int id = (unsigned int)PoolTypeId::MAX; id <= (unsigned int)PoolTypeId::AVERAGE; id++)
		{
			if (to_string((PoolTypeId)id) == str_normalized)
				return (PoolTypeId)id;
		}

		return PoolTypeId::UNKNOWN;
	}
}
