//Copyright (c) 2023 Denys Dragunov, dragunovdenis@gmail.com
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

#include "StopWatch.h"
#include <sstream>

namespace DeepLearning
{
	StopWatch::StopWatch()
	{
		_start = std::chrono::high_resolution_clock::now();
	}

	void StopWatch::reset()
	{
		_start = std::chrono::high_resolution_clock::now();
		_stopped = false;
	}

	void StopWatch::stop()
	{
		_stopped = true;
		_end = std::chrono::high_resolution_clock::now();
	}

	auto StopWatch::get_duration() const
	{
		return (_stopped ? _end :
			std::chrono::high_resolution_clock::now()) - _start;
	}

	long long StopWatch::elapsed_time_in_milliseconds() const
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(get_duration()).count();
	}

	std::string StopWatch::elapsed_time_hh_mm_ss() const
	{
		std::stringstream ss;

		ss << std::chrono::hh_mm_ss(get_duration());

		return ss.str();
	}

	bool StopWatch::is_stopped() const
	{
		return _stopped;
	}
}

