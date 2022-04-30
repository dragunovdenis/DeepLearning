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

namespace DeepLearning
{
	void BasicCollection::add(const BasicCollection& collection)
	{
		if (size() != collection.size())
			throw std::exception("Collections must be of the same size");

		std::transform(begin(), end(), collection.begin(), begin(), [](const auto& x, const auto& y) { return x + y; });
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
		return std::abs(*std::max_element(begin(), end(), [](const auto& x, const auto& y) { return std::abs(x) < std::abs(y); }));
	}

	Real BasicCollection::sum(const std::function<Real(Real)>& transform_operator) const
	{
		return std::accumulate(begin(), end(), Real(0), [&transform_operator](const auto& sum, const auto& x) { return sum + transform_operator(x); });
	}

	void BasicCollection::fill(const Real& val)
	{
		std::fill(begin(), end(), val);
	}

	bool BasicCollection::empty() const
	{
		return size() == 0;
	}

	void BasicCollection::hadamard_prod_in_place(const BasicCollection& collection)
	{
		if (size() != collection.size())
			throw std::exception("Inconsistent input");

		std::transform(begin(), end(), collection.begin(), begin(), [](const auto& x, const auto& y) { return x * y; });
	}

	std::size_t BasicCollection::max_element_id(const std::function<bool(Real, Real)>& comparer) const
	{
		const auto id = std::max_element(begin(), end(), comparer) - begin();
		return static_cast<std::size_t>(id);
	}

	Real& BasicCollection::operator [](const std::size_t& id) { return begin()[id]; };

	const Real& BasicCollection::operator [](const std::size_t& id) const { return begin()[id]; };
}