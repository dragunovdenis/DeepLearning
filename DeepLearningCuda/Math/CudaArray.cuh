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

#include <cstddef>
#include <vector>
#include "CudaUtils.h"

namespace DeepLearning
{
	/// <summary>
	/// An array stored in the CUDA device memory
	/// </summary>
	template <class T>
	class CudaArray
	{
		T* _data{};
		std::size_t _size{};
		std::size_t _capacity{};

		/// <summary>
		/// Abandons (not releases) the allocated resources
		/// </summary>
		void abandon_resources()
		{
			_data = nullptr;
			_size = 0;
			_capacity = 0;
		}

		/// <summary>
		/// Frees memory allocated by the array
		/// </summary>
		void free()
		{
			if (_data != nullptr)
				CudaUtils::cuda_free(_data);

			abandon_resources();
		}
	public:

		/// <summary>
		/// Ensures that the array contains the given number of elements
		/// </summary>
		void resize(const std::size_t& new_size)
		{
			if (_capacity < new_size)
			{
				free();
				_data = CudaUtils::cuda_allocate<T>(new_size);
				_capacity = new_size;
			}

			_size = new_size;
		}

		/// <summary>
		/// Returns number of elements in the array
		/// </summary>
		std::size_t size() const { return _size; }

		/// <summary>
		/// Access to the first element of the array
		/// </summary>
		T* begin()
		{
			return _data;
		}

		/// <summary>
		/// Access to the first element of the array (constant version)
		/// </summary>
		const T* begin() const 
		{
			return _data;
		}

		/// <summary>
		/// Access to the element "past the last"
		/// </summary>
		T* end()
		{
			return _data + _size;
		}

		/// <summary>
		/// Access to the element "past the last" (constant version)
		/// </summary>
		const T* end() const 
		{
			return _data + _size;
		}

		/// <summary>
		/// Default constructor
		/// </summary>
		CudaArray() = default;

		/// <summary>
		/// Creates an instance with the given number of allocated elements
		/// </summary>
		CudaArray(const std::size_t& size)
		{
			resize(size);
		}

		/// <summary>
		/// Copy constructor
		/// </summary>
		CudaArray(const CudaArray<T>& arr)
		{
			resize(arr.size());
			CudaUtils::cuda_copy_device2device(_data, arr._data, size());
		}

		/// <summary>
		/// Move constructor
		/// </summary>
		CudaArray(CudaArray<T>&& arr) noexcept : _data(arr._data), _size(arr._size), _capacity(arr._capacity)
		{
			arr.abandon_resources();
		}

		/// <summary>
		/// Copy-assignment operator
		/// </summary>
		CudaArray<T>& operator =(const CudaArray<T>& arr)
		{
			if (this != &arr)
			{
				resize(arr.size());
				CudaUtils::cuda_copy_device2device(_data, arr._data, size());
			}

			return *this;
		}

		/// <summary>
		/// Move-assignment operator
		/// </summary>
		CudaArray<T>& operator =(CudaArray<T>&& arr) noexcept
		{
			if (this != &arr)
			{
				free();
				_data = arr._data;
				_size = arr._size;
				_capacity = arr._capacity;

				arr.abandon_resources();
			}

			return *this;
		}

		/// <summary>
		/// Converts the array to std-vector
		/// </summary>
		std::vector<T> to_stdvector() const
		{
			std::vector<T> result(size());
			CudaUtils::cuda_copy_device2host(result.data(), _data, size());

			return result;
		}

		/// <summary>
		/// Destructor
		/// </summary>
		~CudaArray()
		{
			free();
		}
	};
}