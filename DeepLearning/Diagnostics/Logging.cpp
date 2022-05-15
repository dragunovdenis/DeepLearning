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

#include "Logging.h"
#include <fstream>
#include <iomanip>
#include <exception>

namespace DeepLearning::Logging
{
	void log_as_table(const RealMemHandleConst& memory, const std::size_t& row_cnt, const std::size_t& col_cnt, const std::filesystem::path& filename)
	{
		if (row_cnt * col_cnt != memory.size())
			throw std::exception("Inconsistent input data");

		std::ofstream file(filename);

		if (!file)
			throw std::exception("Failed to create file");

		const auto num_width = 3;
		const auto place_width = num_width + 4;


		for (auto row_id = 0ull; row_id < row_cnt; row_id++)
		{
			for (auto col_id = 0ull; col_id < col_cnt; col_id++)
			{
				const auto val = memory[row_id * col_cnt + col_id];
				file << std::left << std::setw(place_width) << std::setfill(' ');

				if (val == Real(0))
					file << std::defaultfloat << val;
				else
					file << std::fixed << std::setprecision(num_width) << val;
			}

			file << '\n';
		}
	}

	void make_path(const std::filesystem::path& path)
	{
		if (!std::filesystem::is_directory(path) && !std::filesystem::create_directories(path))
			throw std::exception("Can't create the path");
	}

}