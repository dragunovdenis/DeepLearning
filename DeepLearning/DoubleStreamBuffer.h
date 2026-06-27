#pragma once
#include <fstream>
#include <ios>

namespace DeepLearning
{
	/// <summary>
	/// Custom stream buffer that writes to both console and a log file.
	/// </summary>
	class DoubleStreamBuffer : public std::streambuf
	{
		std::streambuf* _console_buffer;
		std::ofstream _log_file;

	protected:
		/// </inheritdoc/>
		int overflow(int c) override;

		/// </inheritdoc/>
		int sync() override;

	public:
		/// <summary>
		/// Constructor.
		/// </summary>
		DoubleStreamBuffer(std::streambuf* console_buf, const std::string& log_file_path);

		/// <summary>
		/// Destructor.
		/// </summary>
		~DoubleStreamBuffer() override;

		/// <summary>
		/// Returns true if the log file is open, false otherwise.
		/// </summary>
		bool is_open() const;
	};
}
