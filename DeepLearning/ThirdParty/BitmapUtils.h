#pragma once
#include <stdio.h>

namespace ThirdParty
{
	const int BYTES_PER_PIXEL = 3; /// red, green, & blue

	/// <summary>
	/// Save given image to disk in BMP format
	/// </summary>
	/// <param name="image">Collection of bytes to save. Row-wise order.
	/// First row is the bottommost row of the image</param>
	/// <param name="height">Height of the image</param>
	/// <param name="width">Width of the image</param>
	/// <param name="imageFileName">Name of file to save</param>
	void SaveBitmapImage(const unsigned char* image, const int height, const int width, const char* imageFileName);
}

