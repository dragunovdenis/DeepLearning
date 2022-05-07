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

#include "../defs.h"

namespace DeepLearning
{
	/// <summary>
	/// A class that facilitates work with memory allocated elsewhere 
	/// </summary>
	template <class T>
	class MemHandleBase
	{
	protected:
		/// <summary>
		/// Number of elements of type "T" in the handled memory
		/// </summary>
		std::size_t _size{};

		/// <summary>
		/// Pointer to the handled memory
		/// </summary>
		T* _data{};
	public:
		std::size_t size() const
		{
			return _size;
		}

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="data">Pointer to the data to be handled</param>
		/// <param name="size">Number of elements in handled memory</param>
		MemHandleBase(T* const data, const std::size_t& size) : _size(size), _data(data)
		{}
	};

	/// <summary>
	/// Handle providing read only access to the handled memory
	/// </summary>
	template <class T>
	class MemHandleConst: public MemHandleBase<const T>
	{
	public:
		/// <summary>
		/// Read-only subscript operator
		/// </summary>
		const T& operator [](const std::size_t& id) const
		{
			return this->_data[id];
		}

		/// <summary>
		/// Constructor
		/// </summary>
		MemHandleConst(const T* const data, const std::size_t& size) : MemHandleBase<const T>(data, size) {}
	};

	/// <summary>
	/// Handle providing full access to the handled memory
	/// </summary>
	template <class T>
	class MemHandle : public MemHandleBase<T>
	{
	public:
		/// <summary>
		/// Read-only subscript operator
		/// </summary>
		const T& operator [](const std::size_t& id) const
		{
			return this->_data[id];
		}

		/// <summary>
		/// Subscript operator
		/// </summary>
		T& operator [](const std::size_t& id)
		{
			return this->_data[id];
		}

		/// <summary>
		/// Constructor
		/// </summary>
		MemHandle(T* const data, const std::size_t& size) : MemHandleBase<T>(data, size) {}
	};

	using RealMemHandle = MemHandle<Real>;
	using RealMemHandleConst = MemHandleConst<Real>;
}
