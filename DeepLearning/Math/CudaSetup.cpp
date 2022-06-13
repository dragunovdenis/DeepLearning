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

namespace DeepLearning
{
	CudaSetup CudaSetup::_instance = CudaSetup();

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

		if (device_best_score_id >= 0)
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

	unsigned int CudaSetup::threads_per_block()
	{
		return static_cast<unsigned int>(_instance._device_props.maxThreadsPerBlock);
	}

	unsigned int CudaSetup::calc_blocks(const std::size_t& items_cnt)
	{
		return static_cast<unsigned int>((items_cnt + threads_per_block() - 1) / threads_per_block());
	}
}