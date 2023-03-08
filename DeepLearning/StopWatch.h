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

#pragma once
#include <string>
#include <chrono>

namespace DeepLearning
{
	/// <summary>
	/// Functionality to measure elapsed time within the program flow
	/// </summary>
	class StopWatch
	{
		std::chrono::time_point<std::chrono::steady_clock> _start{};
		std::chrono::time_point<std::chrono::steady_clock> _end{};
		bool _stopped{};

		/// <summary>
		/// Returns "current" duration
		/// </summary>
		[[nodiscard]] auto get_duration() const;
	public:
		/// <summary>
		/// Constructor. Automatically starts elapsed-time measurement
		/// </summary>
		StopWatch();

		/// <summary>
		/// Resets time measurement
		/// </summary>
		void reset();

		/// <summary>
		/// Stops the watch
		/// </summary>
		void stop();

		/// <summary>
		///Returns elapsed time in milliseconds;
		/// </summary>
		[[nodiscard]] long long elapsed_time_in_milliseconds() const;

		/// <summary>
		/// Converts elapsed time to `hh:mm:ss` string representation
		/// </summary>
		[[nodiscard]] std::string elapsed_time_hh_mm_ss() const;

		/// <summary>
		/// Returns "true" if the stop-watch is suspended (call `reset` to resume)
		/// </summary>
		[[nodiscard]] bool is_stopped() const;
	};
}
