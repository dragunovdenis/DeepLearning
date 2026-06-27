#include "LogRedirector.h"

namespace DeepLearning
{
	LogRedirector::LogRedirector(const std::string& log_file_path)
		: _original_buffer(std::cout.rdbuf())
		, _double_buffer(nullptr)
	{
		_double_buffer = new DoubleStreamBuffer(_original_buffer, log_file_path);
		if (_double_buffer->is_open())
		{
			std::cout.rdbuf(_double_buffer);
		}
	}

	LogRedirector::~LogRedirector()
	{
		if (_double_buffer)
		{
			// Restore the original buffer
			std::cout.rdbuf(_original_buffer);
			delete _double_buffer;
		}
	}

	bool LogRedirector::is_active() const
	{
		return _double_buffer && _double_buffer->is_open();
	}
}
