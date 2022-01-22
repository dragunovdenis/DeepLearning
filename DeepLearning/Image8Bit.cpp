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

#include "Image8Bit.h"
#include "ThirdParty/BitmapUtils.h"
#include "Math/DenseVector.h"
using namespace ThirdParty;

namespace DeepLearning
{
	Image8Bit::Image8Bit(const std::size_t height, std::size_t width, char* data)
	{
		_height = height;
		_width = width;
		_pixels.resize(_height * _width);

		std::memcpy(_pixels.data(), data, _pixels.size());
	}

	Image8Bit::Image8Bit(const std::size_t height, std::size_t width, std::istream& stream)
	{
		_height = height;
		_width = width;
		_pixels.resize(_height * _width);

		stream.read((char*)_pixels.data(), _pixels.size());
	}

	void Image8Bit::SaveToBmp(const std::filesystem::path& path) const
	{
		std::vector<unsigned char> image(_height * _width * BYTES_PER_PIXEL);

		int i, j;
		for (i = 0; i < _height; i++) {
			for (j = 0; j < _width; j++) {
				//In "bmp" format the bottom row must be the first row of the file.
				//So we need to access our pixel array in a "row-inverted" order.
				//Also we need to invert intensities to have black numbers on white background
				const auto pixel_palue_inverted = 255 - _pixels[(_height - i - 1) * _width + j];
				image[3 * (i * _width + j)] = pixel_palue_inverted;
				image[3 * (i * _width + j) + 1] = pixel_palue_inverted;
				image[3 * (i * _width + j) + 2] = pixel_palue_inverted;
			}
		}

		SaveBitmapImage((unsigned char*)image.data(), _height, _width, path.u8string().c_str());
	}

	Image8Bit::operator DenseVector() const
	{
		return DenseVector(_pixels);
	}
}