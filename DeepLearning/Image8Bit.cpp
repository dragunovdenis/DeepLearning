#include "Image8Bit.h"

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


}