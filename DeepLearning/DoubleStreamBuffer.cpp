
#include "DoubleStreamBuffer.h"

namespace DeepLearning
{
	int DoubleStreamBuffer::overflow(int c)
	{
		if (c == EOF)
			return !EOF;

		// Write to console
		if (_console_buffer->sputc(static_cast<char>(c)) == EOF)
			return EOF;

		// Write to log file
		if (_log_file.is_open())
		{
			_log_file.put(static_cast<char>(c));
			// Flush on newline for crash safety
			if (c == '\n')
				_log_file.flush();
		}

		return c;
	}

	int DoubleStreamBuffer::sync()
	{
		_console_buffer->pubsync();
		if (_log_file.is_open())
			_log_file.flush();
		return 0;
	}

	DoubleStreamBuffer::DoubleStreamBuffer(std::streambuf* console_buf, const std::string& log_file_path)
		: _console_buffer(console_buf)
	{
		_log_file.open(log_file_path, std::ios::out | std::ios::app);
		if (_log_file.is_open())
		{
			// Enable automatic flushing
			_log_file << std::unitbuf;
		}
	}

	DoubleStreamBuffer::~DoubleStreamBuffer()
	{
		if (_log_file.is_open())
		{
			_log_file.flush();
			_log_file.close();
		}
	}

	bool DoubleStreamBuffer::is_open() const
	{
		return _log_file.is_open();
	}
}
