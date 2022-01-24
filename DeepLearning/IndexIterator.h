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
#include <iterator>

namespace DeepLearning
{
	/// <summary>
	/// A forward iterator allowing to traverse integer indices
	/// Can be used, for example, as a substitution to "zip" iterators,
	/// allowing to synchronize access to multiple containers by means of element indices
	/// </summary>
	template <class T>
	struct IndexIterator
	{
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = T;
		using pointer = const T*;
		using reference = const T&;

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="index">Initial index of the iterator</param>
		IndexIterator(const value_type index) :_index(index) {}

		/// <summary>
		/// Dereference operator
		/// </summary>
		reference operator *() const { return _index; }

		/// <summary>
		/// Reference operator
		/// </summary>
		pointer operator ->() const { return &_index; }

		/// <summary>
		/// Prefix increment
		/// </summary>
		IndexIterator& operator++() {
			_index++;
			return *this;
		}

		/// <summary>
		/// Postfix increment
		/// </summary>
		IndexIterator operator ++(int) {
			auto temp = *this;
			++(*this);
			return temp;
		}

		friend bool operator== (const IndexIterator& a, const IndexIterator& b) { return a._index == b._index; };
		friend bool operator!= (const IndexIterator& a, const IndexIterator& b) { return a._index != b._index; };

	private:

		value_type _index;
	};
}
