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

#include "BitmapUtils.h"
#include <exception>

//The implementation below was copied (with some slight modifications) from (answer by Minhas Kamal)
//https://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries

namespace ThirdParty
{
    const int FILE_HEADER_SIZE = 14;
    const int INFO_HEADER_SIZE = 40;

    unsigned char* createBitmapFileHeader(const std::size_t height, const std::size_t stride)
    {
        const auto fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

        static unsigned char fileHeader[] = {
            0,0,     /// signature
            0,0,0,0, /// image file size in bytes
            0,0,0,0, /// reserved
            0,0,0,0, /// start of pixel array
        };

        fileHeader[0] = (unsigned char)('B');
        fileHeader[1] = (unsigned char)('M');
        fileHeader[2] = (unsigned char)(fileSize);
        fileHeader[3] = (unsigned char)(fileSize >> 8);
        fileHeader[4] = (unsigned char)(fileSize >> 16);
        fileHeader[5] = (unsigned char)(fileSize >> 24);
        fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

        return fileHeader;
    }

    unsigned char* createBitmapInfoHeader(const std::size_t height, const std::size_t width)
    {
        static unsigned char infoHeader[] = {
            0,0,0,0, /// header size
            0,0,0,0, /// image width
            0,0,0,0, /// image height
            0,0,     /// number of color planes
            0,0,     /// bits per pixel
            0,0,0,0, /// compression
            0,0,0,0, /// image size
            0,0,0,0, /// horizontal resolution
            0,0,0,0, /// vertical resolution
            0,0,0,0, /// colors in color table
            0,0,0,0, /// important color count
        };

        infoHeader[0] = (unsigned char)(INFO_HEADER_SIZE);
        infoHeader[4] = (unsigned char)(width);
        infoHeader[5] = (unsigned char)(width >> 8);
        infoHeader[6] = (unsigned char)(width >> 16);
        infoHeader[7] = (unsigned char)(width >> 24);
        infoHeader[8] = (unsigned char)(height);
        infoHeader[9] = (unsigned char)(height >> 8);
        infoHeader[10] = (unsigned char)(height >> 16);
        infoHeader[11] = (unsigned char)(height >> 24);
        infoHeader[12] = (unsigned char)(1);
        infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

        return infoHeader;
    }

    void SaveBitmapImage(const unsigned char* image, const std::size_t height, const std::size_t width, const char* imageFileName)
    {
        const auto widthInBytes = width * BYTES_PER_PIXEL;

        const unsigned char padding[3] = { 0, 0, 0 };
        const int paddingSize = (4 - (widthInBytes) % 4) % 4;

        const auto stride = (widthInBytes)+paddingSize;

        FILE* imageFile;
        const errno_t returnValue = fopen_s(&imageFile, imageFileName, "wb");

        if (imageFile == nullptr || returnValue != 0)
            throw std::exception("Can't create a file");

        const unsigned char* fileHeader = createBitmapFileHeader(height, stride);
        fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

        const unsigned char* infoHeader = createBitmapInfoHeader(height, width);
        fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

        for (std::size_t i = 0; i < height; i++) {
            fwrite(image + (i * widthInBytes), BYTES_PER_PIXEL, width, imageFile);
            fwrite(padding, 1, paddingSize, imageFile);
        }

        fclose(imageFile);
    }
}