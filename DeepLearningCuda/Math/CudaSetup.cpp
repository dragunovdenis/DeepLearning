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

#include "CudaSetup.h"
#include <exception>

namespace DeepLearning
{
	/// <summary>
	/// Specify thread local storage for the instance of "CudaSetup" in order for its constructor
	/// is called in each host thread so that cudaSetDevice(...) is called in each thread too;
	/// this does not matter as long as we decide to compute on "0" device, but otherwise can result
	/// in a hard-to-diagnose bugs
	/// </summary>
	thread_local CudaSetup CudaSetup::_instance = CudaSetup();

	/// <summary>
	/// Returns ID of the current CUDA device in the current thread.
	/// </summary>
	int get_current_device()
	{
		int result;

		if (cudaGetDevice(&result) != cudaSuccess)
			throw std::exception("Can't get current CUDA device");

		return result;
	}

	CudaSetup::CudaSetup()
	{
		int deviceCount = -1;
		_status = cudaGetDeviceCount(&deviceCount);

		if (_status != cudaSuccess)
			return;

		if (deviceCount <= 0)
		{
			_status = cudaErrorNoDevice;
			return;
		}

		int device_best_score_id = -1;

		for (auto device_id = 0; device_id < deviceCount; device_id++)
		{
			cudaDeviceProp dev_props;
			_status = cudaGetDeviceProperties(&dev_props, device_id);

			if (_status != cudaSuccess)
				continue;

			if (device_best_score_id < 0 || dev_props.multiProcessorCount > _device_props.multiProcessorCount)
			{
				_device_props = dev_props;
				device_best_score_id = device_id;
			}
		}

		if (device_best_score_id >= 0 && get_current_device() != device_best_score_id)
			cudaSetDevice(device_best_score_id);
	}

	bool CudaSetup::is_successful()
	{
		return _instance._status == cudaSuccess;
	}

	cudaError_t CudaSetup::status()
	{
		return _instance._status;
	}

	unsigned int CudaSetup::max_threads_per_block()
	{
		return static_cast<unsigned int>(_instance._device_props.maxThreadsPerBlock);
	}

	std::size_t CudaSetup::shared_memory_per_block()
	{
		return _instance._device_props.sharedMemPerBlock;
	}

	unsigned int CudaSetup::calc_blocks(const std::size_t& total_threads, const unsigned int threads_per_block)
	{
		const auto threads_per_block_to_use = (threads_per_block == 0) ? max_threads_per_block() : threads_per_block;

		return static_cast<unsigned int>((total_threads + threads_per_block_to_use - 1) / threads_per_block_to_use);
	}
}