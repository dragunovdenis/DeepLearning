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

//more about this in reference 1
#ifdef EXPORT
#define DLL_IMPORT_EXPORT  __declspec(dllexport)   // export DLL information

#else
#define DLL_IMPORT_EXPORT  __declspec(dllimport)   // import DLL information

#endif 

namespace ScottPlotBridge
{
    /// <summary>
    /// Different options to align legend frame on the plot
    /// </summary>
    enum class LegendAlignment : int
    {
        Auto = -1,
        UpperLeft = 0,
        UpperRight = 1,
        UpperCenter = 2,
        MiddleLeft = 3,
        MiddleCenter = 4,
        MiddleRight = 5,
        LowerLeft = 6,
        LowerRight = 7,
        LowerCenter = 8
    };

    /// <summary>
    /// Series of Y coordinates to plot
    /// </summary>
    struct YSeries
    {
        /// <summary>
        /// Collection of "Y" coordinates to plot
        /// </summary>
        std::vector<double> y;

        /// <summary>
        /// Text label of the series
        /// </summary>
        std::string label;

        /// <summary>
        /// Width of the line used when plotting
        /// </summary>
        int line_width = 1;
    };

    /// <summary>
    /// Data structure to contain information needed to plot multiple series of point
    /// </summary>
    struct SeriesData
    {
        /// <summary>
        /// X coordinates (shared among all the series)
        /// </summary>
        std::vector<double> x;

        /// <summary>
        /// Collection of "Y" coordinates (each series in the collection must have the same number of elements equal to the number of "X" coordinates)
        /// </summary>
        std::vector<YSeries> series;
    };

    /// <summary>
    /// Plots given set of 2D points represented with their x,y coordinates to the given "png" file on disk
    /// </summary>
    /// <param name="x">x-coordinates</param>
    /// <param name="y">y-coordinates</param>
    /// <param name="plot_width">Plot image width (pixels)</param>
    /// <param name="plot_height">Plot image height (pixels)</param>
    /// <param name="plotFileName">Name of the file where to save the plotted image</param>
    /// <returns></returns>
    DLL_IMPORT_EXPORT void PlotFunction(const SeriesData& data, const std::filesystem::path& plotFileName,
        const int plot_width = 600, const int plot_height = 800, const std::string& x_label = "X",
        const std::string& y_label = "Y", const std::string& title = "Title",
        const LegendAlignment legend_alignment = LegendAlignment::Auto);
}
