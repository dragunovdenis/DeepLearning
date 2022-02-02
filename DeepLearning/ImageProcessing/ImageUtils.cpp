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

#include "ImageUtils.h"
#include "../Image8Bit.h"
#include <cmath>

namespace DeepLearning::ImageUtils
{
	/// <summary>
	/// Wrapper to access pixels of the given image in a "safe" way.
	/// If the given row or column coordinates are negative or exceed the corresponding
	/// dimensions of the image the method returns the given "default pixel" value
	/// </summary>
	template <class Image>
	Image::pixel_t access_pixel_safely(const Image& image,
		const long long& row_id, const long long& col_id,
		const typename Image::pixel_t& default_pixel = Image::pixel_t(0))
	{
		if (row_id < 0 || static_cast<long long>(image.height()) <= row_id ||
			col_id < 0 || static_cast<long long>(image.width()) <= col_id)
			return default_pixel;

		return image(row_id, col_id);
	}

	template <class Image>
	Image transform(const Image& image, const MatrixAffine2d<Real>& transformation)
	{
		const auto inv_trans = transformation.inverse();

		//create an "empty" image of the same size as the input image
		auto result = Image(image.height(), image.width());

		for (std::size_t row_id = 0; row_id < image.height(); row_id++)
			for (std::size_t col_id = 0; col_id < image.width(); col_id++)
			{
				//point representing center of the current pixel (we assume that pixel is a square with side 1)
				const auto center_pt = Vector2d<Real>{ Real(col_id) + Real(0.5), Real(row_id) + Real(0.5)};
				const auto center_pt_transformed = inv_trans * center_pt;

				//find row and column indices of a base pixel that "contains" top left corner of the rotated pixel 
				const auto tl_pt_transformed = Vector2d<Real>{ center_pt_transformed.x - Real(0.5),
					center_pt_transformed.y - Real(0.5) };
				const auto col_id_base = static_cast<long long>(std::floor(tl_pt_transformed.x));
				const auto row_id_base = static_cast<long long>(std::floor(tl_pt_transformed.y));

				result(row_id, col_id) = static_cast<Image::pixel_t>(access_pixel_safely(image, row_id_base, col_id_base) *
					(col_id_base + 1 - tl_pt_transformed.x) * (row_id_base + 1 - tl_pt_transformed.y) +
					access_pixel_safely(image, row_id_base + 1, col_id_base) *
					(col_id_base + 1 - tl_pt_transformed.x) * (tl_pt_transformed.y - row_id_base) +
					access_pixel_safely(image, row_id_base, col_id_base + 1) *
					(tl_pt_transformed.x - col_id_base) * (row_id_base + 1 - tl_pt_transformed.y) +
					access_pixel_safely(image, row_id_base + 1, col_id_base + 1) *
					(tl_pt_transformed.x - col_id_base) * (tl_pt_transformed.y - row_id_base));
			}

		return result;
	}

	template Image8Bit transform(const Image8Bit& image, const MatrixAffine2d<Real>& transformation);
}