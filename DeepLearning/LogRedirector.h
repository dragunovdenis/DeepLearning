#pragma once
#include <iostream>

#include "DoubleStreamBuffer.h"

namespace DeepLearning
{
	/// <summary>
	/// Functionality, that when enabled, automatically
	/// duplicates all output to std::cout to a log file.
	/// </summary>
	class LogRedirector
	{
		std::streambuf* _original_buffer;
		DoubleStreamBuffer* _double_buffer;

	public:
		/// <summary>
		/// Constructor.
		/// </summary>
		LogRedirector(const std::string& log_file_path);

		/// <summary>
		/// Destructor.
		/// </summary>
		~LogRedirector();

		/// <summary>
		/// Returns true if the log redirection is active (alive), false otherwise.
		/// </summary>
		bool is_active() const;
	};
}
