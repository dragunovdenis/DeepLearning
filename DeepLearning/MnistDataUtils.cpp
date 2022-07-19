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

#include "MnistDataUtils.h"
#include "Math/Vector.h"
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

		if (!std::filesystem::exists(path))
			throw std::exception((std::string("File does not exist : ") + path.string()).c_str());

		if (file.fail())
			throw std::exception("Can't open file");

		file.seekg(0, file.end);
		const int length = static_cast<int>(file.tellg());
		file.seekg(0, file.beg);

		const int image_height = 28;
		const int image_width = 28;

		if (length != 16/*size of the expected headed in bytes*/ +
			image_height * image_width * expected_image_count)
			throw std::exception("Unexpected size of the file.");

		int temp;
		file.read((char*)&temp, sizeof(int)); // Read the magic number and discard it
		if (reverseInt(temp) != 2051)
			throw std::exception("Unexpected magic number value.");

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
			result.push_back(Image8Bit(image_height, image_width, file));

		return result;
	}

	std::vector<Tensor> MnistDataUtils::read_labels(const std::filesystem::path& path,
		const std::size_t expected_label_count)
	{
		std::ifstream file;
		file.open(path, std::ios::in | std::ios::binary);

		if (file.fail())
			throw std::exception("Can't open file");

		file.seekg(0, file.end);
		const int length = static_cast<int>(file.tellg());
		file.seekg(0, file.beg);

		if (length != 8/*size of the expected headed in bytes*/ + expected_label_count)
			throw std::exception("Unexpected size of the file.");

		int temp;
		file.read((char*)&temp, sizeof(int)); // Read the magic number and discard it
		if (reverseInt(temp) != 2049)
			throw std::exception("Unexpected magic number value.");

		file.read((char*)&temp, sizeof(int)); // Read number of images
		//sanity check
		if (expected_label_count != reverseInt(temp))
			throw std::exception("Unexpected number of labels in the file.");

		//The header is over
		std::vector<Tensor> result(expected_label_count, Tensor(1, 1, 10));

		unsigned char u_byte;
		for (std::size_t label_id = 0; label_id < expected_label_count; label_id++)
		{
			file.read((char*)&u_byte, sizeof(unsigned char));
			if (u_byte > 9)
				throw std::exception("Unexpected label value.");

			result[label_id](0, 0, u_byte) = Real(1);
		}

		return result;
	}

	std::vector<Tensor> MnistDataUtils::scale_images(const std::vector<Image8Bit>& images, const bool flatten_images, const Real& max_value)
	{
		if (images.empty())
			return std::vector<Tensor>();

		std::vector<Tensor> result(images.begin(), images.end());

		const auto image_size = result.begin()->size();
		if (!std::all_of(result.begin(), result.end(), [=](const auto& im) { return im.size() == image_size; }))
			throw std::exception("Images are supposed to be of same size");

		const auto scale_factor = max_value / 256;

		for (auto& result_item : result)
			result_item *= scale_factor;

		if (!flatten_images)
		{
			const auto shape = Index3d{ 1ull, images.begin()->height(), images.begin()->width() };
			for (auto& result_item : result)
				result_item.reshape(shape);
		}

		return result;
	}

	/// <summary>
	/// Converts given "host" collection to "device" collection
	/// </summary>
	std::vector<CudaTensor> to_device(const std::vector<Tensor>& data)
	{
		std::vector<CudaTensor> result(data.size());

		for (auto data_id = 0ull; data_id < data.size(); data_id++)
			result[data_id].assign(data[data_id]);

		return result;
	}

	template<>
	std::tuple<std::vector<typename CpuDC::tensor_t>, std::vector<typename CpuDC::tensor_t>> MnistDataUtils::load_labeled_data<CpuDC>(
		const std::filesystem::path& data_path, const std::filesystem::path& labels_path,
		const std::size_t expected_items_count, const bool flatten_images, const Real& max_value)
	{
		const auto images = MnistDataUtils::read_images(data_path, expected_items_count);
		const auto images_scaled = scale_images(images, flatten_images, max_value);
		const auto labels = MnistDataUtils::read_labels(labels_path, expected_items_count);

		return std::make_tuple(images_scaled, labels);
	}

	template<>
	std::tuple<std::vector<typename GpuDC::tensor_t>, std::vector<typename GpuDC::tensor_t>> MnistDataUtils::load_labeled_data<GpuDC>(
		const std::filesystem::path& data_path, const std::filesystem::path& labels_path,
		const std::size_t expected_items_count, const bool flatten_images, const Real& max_value)
	{
		const auto [images_scaled, labels] = MnistDataUtils::load_labeled_data<CpuDC>(data_path, labels_path,	expected_items_count, flatten_images, max_value);
		return std::make_tuple(to_device(images_scaled), to_device(labels));
	}
}