#pragma once
#include <vector>
#include <istream>
#include <filesystem>

namespace DeepLearning
{
	class DenseVector;

	/// <summary>
	/// Representation of a single channel 8-bit image
	/// </summary>
	class Image8Bit
	{
		std::vector<unsigned char> _pixels{};
		std::size_t _height{};
		std::size_t _width{};
	public:
		/// <summary>
		/// Default constructor
		/// </summary>
		Image8Bit() = default;

		/// <summary>
		/// Constructor
		/// </summary>
		Image8Bit(const std::size_t height, std::size_t width, char* data);

		/// <summary>
		/// Constructor (from a stream)
		/// </summary>
		Image8Bit(const std::size_t height, std::size_t width, std::istream& stream);

		/// <summary>
		/// Saves the image to disk in "bmp" format
		/// </summary>
		void SaveToBmp(const std::filesystem::path& path) const;

		/// <summary>
		/// Implicit conversion operator
		/// </summary>
		operator DenseVector() const;
	};
}
