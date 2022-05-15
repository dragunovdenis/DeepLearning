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
#include "../Memory/MemHandle.h"
#include <filesystem>

namespace DeepLearning::Logging
{
	/// <summary>
	/// Logs the handled memory of "real" elements to a text file on disk in a form of rectangular table with the given number of rows and columns
	/// </summary>
	/// <param name="memory">Handle to the memory that needs to be logged</param>
	/// <param name="row_cnt">Expected number of tows in the logged table</param>
	/// <param name="col_cnt">Expected number of columns in the logged table</param>
	/// <param name="filename">Full path to the text file to log the table</param>
	void log_as_table(const RealMemHandleConst& memory, const std::size_t& row_cnt, const std::size_t& col_cnt, const std::filesystem::path& filename);

	/// <summary>
	/// Creates path of sub-folders if the path does not exist
	/// </summary>
	/// <param name="path">Path to be ensured</param>
	void make_path(const std::filesystem::path& path);
}

