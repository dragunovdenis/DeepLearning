#include "CppUnitTest.h"
#include <Image8Bit.h>
#include <ImageProcessing/ImageUtils.h>
#include <numbers>
#include <array>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(ImageProcessingTest)
	{
		/// <summary>
		/// Draws a cross pattern on the given image according to the given collection of intensities
		/// and the given region boundaries
		/// </summary>
		static void draw_cross_pattern(Image8Bit& image, const std::array<unsigned char, 5>& pattern_intensities,
			const std::size_t row_begin, const std::size_t row_end,
			const std::size_t col_begin, const std::size_t col_end)
		{
			const auto center_row_id = (row_end + row_begin) / 2;
			const auto center_col_id = (col_end + col_begin) / 2;

			for (std::size_t row_id = row_begin; row_id < row_end; row_id++)
				for (std::size_t col_id = col_begin; col_id < col_end; col_id++)
				{
					if (row_id == center_row_id && col_id < center_col_id)
						image(row_id, col_id) = pattern_intensities[3];
					else if (row_id == center_row_id && col_id > center_col_id)
						image(row_id, col_id) = pattern_intensities[1];
					else if (row_id < center_row_id && col_id == center_col_id)
						image(row_id, col_id) = pattern_intensities[0];
					else if (row_id > center_row_id && col_id == center_col_id)
						image(row_id, col_id) = pattern_intensities[2];
					else if (row_id == center_row_id && col_id == center_col_id)
						image(row_id, col_id) = pattern_intensities[4];
					else
						image(row_id, col_id) = 0;
				}
		}

		TEST_METHOD(ImageTransformationTest)
		{
			//Arrange
			std::size_t image_height = 11;
			std::size_t image_width = 21;
			auto image = Image8Bit(image_height, image_width);

			const Image8Bit::pixel_t north_marker  = 50;
			const Image8Bit::pixel_t west_marker   = 100;
			const Image8Bit::pixel_t south_marker  = 150;
			const Image8Bit::pixel_t east_marker   = 200;
			const Image8Bit::pixel_t center_marker = 250;

			draw_cross_pattern(image, { north_marker , east_marker, south_marker, west_marker, center_marker},
				0, image.height(), 0, image.width());

			//Rotation around center of the image
			const auto transformation = MatrixAffine2d<Real>::rotation(Real(std::numbers::pi/2),
				Vector2d<Real>{image_width * Real(0.5), image_height * Real(0.5)});

			//Act
			const auto image_transformed = ImageUtils::transform(image, transformation);

			//We know what should be the result of rotation of the cross pattern by 90 degrees
			//so we can draw the reference image in the following way
			auto image_reference = Image8Bit(image_height, image_width);
			draw_cross_pattern(image_reference, { west_marker, north_marker , east_marker, south_marker, center_marker },
				0, image.height(), (image_width - image_height)/2 , image_width - (image_width - image_height) / 2);

			Assert::IsTrue(image_reference == image_transformed, L"Transformed image differs from the reference one");
		}
	};
}
