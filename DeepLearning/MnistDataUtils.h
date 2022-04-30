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
#include "Image8Bit.h"
#include "Math/Tensor.h"
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
		static std::vector<Tensor> read_labels(const std::filesystem::path& path,
			const std::size_t expected_label_count);
	};
}
