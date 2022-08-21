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
        const std::string& y_label = "Y", const std::string& title = "Title");
}
