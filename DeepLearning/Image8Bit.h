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
#include <vector>
#include <filesystem>

namespace DeepLearning
{
	class Vector;

	/// <summary>
	/// Representation of a single channel 8-bit image
	/// </summary>
	class Image8Bit
	{
	public:
		using pixel_t = unsigned char;

	private:
		std::vector<pixel_t> _pixels{};
		std::size_t _height{};
		std::size_t _width{};
	public:

		/// <summary>
		/// Read access to the height parameter of the image
		/// </summary>
		[[nodiscard]] std::size_t height() const;

		/// <summary>
		/// Read access to the width parameter of the image
		/// </summary>
		[[nodiscard]] std::size_t width() const;

		/// <summary>
		/// Default constructor
		/// </summary>
		Image8Bit() = default;

		/// <summary>
		/// Constructor
		/// </summary>
		Image8Bit(const std::size_t height, std::size_t width, char* data);

		/// <summary>
		/// Constructor
		/// </summary>
		Image8Bit(const std::size_t height, std::size_t width);

		/// <summary>
		/// Constructor (from a stream)
		/// </summary>
		Image8Bit(const std::size_t height, std::size_t width, std::istream& stream);

		/// <summary>
		/// Access to a pixel with the given row and column indices. Check for index boundaries is on the caller.
		/// </summary>
		pixel_t& operator()(const std::size_t row_id, const std::size_t col_id);

		/// <summary>
		/// Access to a pixel with the given row and column indices (constant version). Check for index boundaries is on the caller.
		/// </summary>
		const pixel_t& operator()(const std::size_t row_id, const std::size_t col_id) const;

		/// <summary>
		/// Saves the image to disk in "bmp" format
		/// </summary>
		void SaveToBmp(const std::filesystem::path& path) const;

		/// <summary>
		/// "Equal to" operator
		/// </summary>
		bool operator ==(const Image8Bit& anotherImage) const;

		/// <summary>
		/// "Not equal to" operator
		/// </summary>
		bool operator !=(const Image8Bit& anotherImage) const;

		/// <summary>
		/// Implicit conversion operator
		/// </summary>
		operator Vector() const;
	};
}
