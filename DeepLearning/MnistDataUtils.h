#pragma once
#include "Image8Bit.h"
#include "Math/DenseVector.h"
#include <filesystem>

namespace DeepLearning
{
	/// <summary>
	/// Utility methods to work with the MNIST data (see http://yann.lecun.com/exdb/mnist/)
	/// </summary>
	class MnistDataUtils
	{
	public:
		/// <summary>
		/// Reads MNIST images from a file on disk following the protocol:
		/// [offset] [type]          [value]          [description]
		/// 0000     32 bit integer  0x00000803(2051) magic number
		///	0004     32 bit integer  60000            number of images
		///	0008     32 bit integer  28               number of rows
		///	0012     32 bit integer  28               number of columns
		///	0016     unsigned byte   ??				  pixel
		///	0017     unsigned byte   ??				  pixel
		///	........
		///	xxxx     unsigned byte   ??				  pixel
		/// </summary>
		/// <param name="path">Path to the corresponding file on disk</param>
		/// <param name="expected_image_count">Expected number of images to read</param>
		static std::vector<Image8Bit> read_images(const std::filesystem::path& path,
			const std::size_t expected_image_count);

		/// <summary>
		/// Read MNIST labels from a file on disk following the protocol:
		/// [offset] [type]          [value]          [description]
		/// 0000     32 bit integer  0x00000801(2049) magic number(MSB first)
		///	0004     32 bit integer  60000            number of items
		///	0008     unsigned byte   ??               label
		///	0009     unsigned byte   ??               label
		///	........
		///	xxxx     unsigned byte   ??               label
		/// </summary>
		/// <param name="path">Path to the corresponding file on disk</param>
		/// <param name="expected_label_count">Expected number of labels to read</param>
		/// <returns></returns>
		static std::vector<DenseVector> read_labels(const std::filesystem::path& path,
			const std::size_t expected_label_count);
	};
}
