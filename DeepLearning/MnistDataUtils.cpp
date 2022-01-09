#include "MnistDataUtils.h"
#include <fstream>

namespace DeepLearning
{
	/// <summary>
	/// Reverses bytes in the given integer
	/// </summary>
	int reverseInt(int i)
	{
		unsigned char c1, c2, c3, c4;

		c1 = i & 255;
		c2 = (i >> 8) & 255;
		c3 = (i >> 16) & 255;
		c4 = (i >> 24) & 255;

		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	}

	std::vector<Image8Bit> MnistDataUtils::read_images(const std::filesystem::path& path,
		const std::size_t expected_image_count)
	{
		std::ifstream file;
		file.open(path, std::ios::in | std::ios::binary);

		if (file.fail())
			throw std::exception("Can't open file");

		file.seekg(0, file.end);
		const int length = file.tellg();
		file.seekg(0, file.beg);

		const int image_height = 28;
		const int image_width = 28;

		if (length != 16/*size of the expected headed in bytes*/ +
			image_height * image_width * expected_image_count)
			throw std::exception("Unexpected size of the file.");

		int temp;
		file.read((char*)&temp, sizeof(int)); // Read the magic number and discard it
		file.read((char*)&temp, sizeof(int)); // Read number of images
		//sanity check
		if (expected_image_count != reverseInt(temp))
			throw std::exception("Unexpected number of images in the file.");

		file.read((char*)&temp, sizeof(int)); // read height of an image
		if (reverseInt(temp) != image_height)
			throw std::exception("Unexpected image height.");

		file.read((char*)&temp, sizeof(int)); // read width of an image
		if (reverseInt(temp) != image_width)
			throw std::exception("Unexpected image width.");

		//The header is over

		std::vector<Image8Bit> result;

		for (std::size_t image_id = 0; image_id < expected_image_count; image_id++)
		{
			result.push_back(Image8Bit(image_height, image_width, file));
		}

		return result;
	}
}