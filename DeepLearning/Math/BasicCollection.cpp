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

#include "BasicCollection.h"
#include <algorithm>
#include <numeric>
#include <exception>
#include <random>
#include <ranges>
#include "AvxAcceleration.h"
#include <atomic>

namespace DeepLearning
{
	static std::atomic<unsigned int> instance_counter{ 0 };
	static std::atomic<std::size_t> allocated_memory_counter{ 0 };

	void BasicCollection::allocate(const std::size_t new_capacity)
	{
		if (_capacity < new_capacity)
		{
			free();
			_data = static_cast<Real*>(std::malloc(new_capacity * sizeof(Real)));
			_capacity = new_capacity;
			allocated_memory_counter.fetch_add(_capacity);
		}
	}

	void BasicCollection::free()
	{
		if (_data != nullptr)
		{
			std::free(_data);
			_data = nullptr;
			allocated_memory_counter.fetch_sub(_capacity);
		}

		_capacity = 0;
	}

	void BasicCollection::abandon_resources()
	{
		_data = nullptr;
		free();
	}

	void BasicCollection::take_over_resources(BasicCollection&& collection)
	{
		if (this == &collection)
			return;

		free();
		_data = collection._data;
		_capacity = collection._capacity;
		collection.abandon_resources();
	}

	BasicCollection::BasicCollection()
	{
		instance_counter.fetch_add(1);
	}

	BasicCollection::~BasicCollection()
	{
		free();
		instance_counter.fetch_sub(1);
	}

	unsigned BasicCollection::get_total_instances_count()
	{
		return instance_counter;
	}

	std::size_t BasicCollection::get_total_allocated_memory()
	{
		return allocated_memory_counter * sizeof(Real);
	}

	std::size_t BasicCollection::get_allocated_memory() const
	{
		return _capacity * sizeof(Real);
	}

	void BasicCollection::add(const BasicCollection& collection)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		std::transform(begin(), end(), collection.begin(), begin(), [](const auto& x, const auto& y) { return x + y; });
	}

	void BasicCollection::add_scaled(const BasicCollection& collection, const Real& scalar)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		std::transform(begin(), end(), collection.begin(), begin(),
			[scalar](const auto& x, const auto& y) { return x + y * scalar; });
	}

	void BasicCollection::scale_and_add_scaled(const Real& scalar_0, const BasicCollection& collection,
		const Real& scalar_1)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		std::transform(begin(), end(), collection.begin(), begin(),
			[scalar_0, scalar_1](const auto& x, const auto& y) { return x * scalar_0 + y * scalar_1; });
	}

	void BasicCollection::scale_and_add(const BasicCollection& collection, const Real& scalar)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		std::transform(begin(), end(), collection.begin(), begin(), [scalar](const auto& x, const auto& y) { return x * scalar + y ; });
	}

	void BasicCollection::sub(const BasicCollection& collection)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		std::transform(begin(), end(), collection.begin(), begin(), [](const auto& x, const auto& y) { return x - y; });
	}

	void BasicCollection::mul(const Real& scalar)
	{
		std::transform(begin(), end(), begin(), [scalar](const auto& x) { return x * scalar; });
	}

	Real BasicCollection::max_abs() const
	{
		if (empty())
			return std::numeric_limits<Real>::signaling_NaN();

		return std::abs(*std::max_element(begin(), end(), [](const auto& x, const auto& y) { return std::abs(x) < std::abs(y); }));
	}

	Real BasicCollection::sum(const std::function<Real(Real)>& transform_operator) const
	{
		return std::accumulate(begin(), end(), Real(0), [&transform_operator](const auto& sum, const auto& x) { return sum + transform_operator(x); });
	}

	Real BasicCollection::sum_of_squares() const
	{
		return std::accumulate(begin(), end(), Real(0), [](const auto& sum, const auto& x) { return sum + x * x; });
	}

	void BasicCollection::fill(const Real& val)
	{
		std::fill(begin(), end(), val);
	}

	void BasicCollection::fill_zero()
	{
		std::memset(begin(), 0, size() * sizeof(Real));
	}

	bool BasicCollection::empty() const
	{
		return size() == 0;
	}

	Real* BasicCollection::begin()
	{
		return _data;
	}

	const Real* BasicCollection::begin() const
	{
		return _data;
	}

	Real* BasicCollection::end()
	{
		return _data + size();
	}

	const Real* BasicCollection::end() const
	{
		return _data + size();
	}

	void BasicCollection::hadamard_prod_in_place(const BasicCollection& collection)
	{
		hadamard_prod(*this, collection);
	}

	void BasicCollection::hadamard_prod(const BasicCollection& op0, const BasicCollection& op1)
	{
		if (size() != op0.size() || size() != op1.size())
			throw std::exception("Inconsistent input");

		std::transform(op0.begin(), op0.end(), op1.begin(), begin(),
			[](const auto& x, const auto& y) { return x * y; });
	}

	void BasicCollection::hadamard_prod_add(const BasicCollection& op0, const BasicCollection& op1)
	{
		if (size() != op0.size() || size() != op1.size())
			throw std::exception("Inconsistent input");

		const auto op0_arr = op0.begin();
		const auto op1_arr = op1.begin();
		auto res_arr = begin();

		for (auto id = 0ull; id < size(); id++)
			res_arr[id] += op0_arr[id] * op1_arr[id];
	}

	Real BasicCollection::dot_product(const BasicCollection& collection) const
	{
		if (size() != collection.size())
			throw std::exception("Inconsistent input data");

#ifdef USE_AVX2
		return Avx::mm256_dot_product(&*begin(), &*collection.begin(), size());
#else
		return std::inner_product(begin(), end(), collection.begin(), Real(0));
#endif // USE_AVX2
	}

	std::size_t BasicCollection::max_element_id(const std::function<bool(Real, Real)>& comparer) const
	{
		const auto id = std::max_element(begin(), end(), comparer) - begin();
		return static_cast<std::size_t>(id);
	}

	Real BasicCollection::max_element(const std::function<bool(Real, Real)>& comparer) const
	{
		if (empty())
			return std::numeric_limits<Real>::quiet_NaN();

		return *std::max_element(begin(), end(), comparer);
	}

	Real& BasicCollection::operator [](const std::size_t& id) { return _data[id]; };

	const Real& BasicCollection::operator [](const std::size_t& id) const { return _data[id]; };

	std::vector<Real> BasicCollection::to_stdvector() const
	{
		return std::vector<Real>(begin(), end());
	}

	RealMemHandleConst BasicCollection::get_handle() const
	{
		return RealMemHandleConst(begin(), size());
	}

	RealMemHandle BasicCollection::get_handle()
	{
		return RealMemHandle(begin(), size());
	}

	void BasicCollection::uniform_random_fill(const Real& min, const Real& max, std::mt19937* seeder)
	{
		if (empty())
			return;

		thread_local std::mt19937 gen(std::random_device{}());
		auto& gen_alias = seeder ? *seeder : gen;
		std::uniform_real_distribution<Real> dist(min, max);
		std::generate(begin(), end(), [&]() {return dist(gen_alias); });
	}

	void BasicCollection::standard_random_fill(const Real& sigma, std::mt19937* seeder)
	{
		if (empty())
			return;

		thread_local std::mt19937 gen(std::random_device{}());
		auto& gen_alias = seeder ? *seeder : gen;
		std::normal_distribution<Real> dist{ 0, sigma < Real(0) ? Real(1) / Real(std::sqrt(size())) : sigma };
		std::generate(begin(), end(), [&]() {return dist(gen_alias); });
	}

	void BasicCollection::init(const InitializationStrategy strategy, std::mt19937* seeder)
	{
		switch (strategy)
		{
			case None: return;
			case FillZero: fill_zero(); return;
			case FillRandomUniform: uniform_random_fill(-1, 1, seeder); return;
			case FillRandomNormal: standard_random_fill(1, seeder); return;
			default:
				throw std::exception("Unknown initialization strategy.");
		}
	}

	bool BasicCollection::is_nan() const
	{
		return std::any_of(begin(), end(), [](const auto& x) { return std::isnan(x); });
	}

	bool BasicCollection::is_inf() const
	{
		return std::any_of(begin(), end(), [](const auto& x) { return std::isinf(x); });
	}
}
