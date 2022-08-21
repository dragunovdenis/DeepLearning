using System;

namespace ScottPlotWrapper
{
    /// <summary>
    /// Series of Y coordinates to plot
    /// </summary>
    public class YSeries
    {
        /// <summary>
        /// Collection of "Y" coordinates to plot
        /// </summary>
        public double[] Y;

        /// <summary>
        /// Text label of the series
        /// </summary>
        public string Label;
    }

    /// <summary>
    /// Data structure to contain information needed to plot multiple series of point
    /// </summary>
    public class SeriesData
    {
        /// <summary>
        /// X coordinates (shared among all the series)
        /// </summary>
        public double[] X;

        /// <summary>
        /// Collection of "Y" coordinates (each series in the collection must have the same number of elements equal to the number of "X" coordinates)
        /// </summary>
        public YSeries[] Series;
    }

    /// <summary>
    /// Functionality for plotting 2D data
    /// </summary>
    public static class ScottPlotter
    {
        /// <summary>
        /// Plots given set of 2D points represented with their x,y coordinates to the given "png" file on disk
        /// </summary>
        /// <param name="x">x-coordinates</param>
        /// <param name="y">y-coordinates</param>
        /// <param name="plotWidth">Plot image width (pixels)</param>
        /// <param name="plotHeight">Plot image height (pixels)</param>
        /// <param name="plotFileName">Name of the file where to save the plotted image</param>
        /// <param name="xLabel">Text label of the horizontal axis</param>
        /// <param name="yLabel">Text label of the vertical axis</param>
        /// <param name="title">Title of the plot</param>
        public static void PlotFunction(SeriesData data, string plotFileName, int plotWidth = 600, int plotHeight = 800, string xLabel = "X", string yLabel = "Y", string title = "Title")
        {
            var plt = new ScottPlot.Plot(plotWidth, plotHeight);
            plt.XLabel(xLabel);
            plt.YLabel(yLabel);

            foreach (var series in data.Series)
            {
                if (data.X.Length != series.Y.Length)
                    throw new Exception("Incompatible input data");
                plt.AddScatter(data.X, series.Y, label: series.Label);
            }

            plt.Title(title); plt.SaveFig(plotFileName);
        }
    }
}
