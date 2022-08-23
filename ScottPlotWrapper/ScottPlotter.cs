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

using System;
using System.Linq;
using System.Collections.Generic;

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

        /// <summary>
        /// Width of the line used when plotting
        /// </summary>
        public int LineWidth = 1;
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
    /// 2D bounding box
    /// </summary>
    class BoundingBox
    {
        /// <summary>
        /// Minimal x
        /// </summary>
        public double MinX { get; private set; } = Double.PositiveInfinity;
        /// <summary>
        /// Maximal x
        /// </summary>
        public double MaxX { get; private set; } = Double.NegativeInfinity;

        /// <summary>
        /// Minimal y
        /// </summary>
        public double MinY { get; private set; } = Double.PositiveInfinity;
        /// <summary>
        /// Maximal y
        /// </summary>
        public double MaxY { get; private set; } = Double.NegativeInfinity;

        /// <summary>
        /// Adds point to the bounding box
        /// </summary>
        public void Add(double x, double y)
        {
            MaxX = Math.Max(MaxX, x);
            MaxY = Math.Max(MaxY, y);

            MinX = Math.Min(MinX, x);
            MinY = Math.Min(MinY, y);
        }
    }

    /// <summary>
    /// Functionality for plotting 2D data
    /// </summary>
    public static class ScottPlotter
    {
        /// <summary>
        /// Enumerates points in the given series data
        /// </summary>
        static IEnumerable<Tuple<double, double>> IteratePoints(SeriesData data)
        {
            foreach (var series in data.Series)
            {
                if (data.X.Length != series.Y.Length)
                    throw new Exception("Incompatible input data");

                for (var i = 0; i < data.X.Length; i++)
                    yield return new Tuple<double, double>(data.X[i], series.Y[i]);
            }
        }

        /// <summary>
        /// Estimates best position for the legend frame based on the location of points
        /// </summary>
        static ScottPlot.Alignment EstimateLegendAlignment(SeriesData data)
        {
            var bb = new BoundingBox();

            foreach (var pt in IteratePoints(data))
                bb.Add(pt.Item1, pt.Item2);

            var middleX = (bb.MaxX + bb.MinX) / 2;
            var middleY = (bb.MaxY + bb.MinY) / 2;

            if (middleX == 0 || middleY == 0)
                return ScottPlot.Alignment.LowerRight;

            //count number of points in each of the four quadrants
            var quadrantCount = new int[2, 2];
            foreach (var pt in IteratePoints(data))
            {
                var quadX = middleX < pt.Item1 ? 1 : 0;
                var quadY = middleY < pt.Item2 ? 1 : 0;
                quadrantCount[quadY, quadX]++;
            }

            var quadMinY = 0;
            var quadMinX = 0;
            var quadMinPts = int.MaxValue;
            for (var quadY = 0; quadY < 2; quadY++)
                for (var quadX = 0; quadX < 2; quadX++)
                {
                    if (quadMinPts > quadrantCount[quadY, quadX])
                    {
                        quadMinY = quadY;
                        quadMinX = quadX;
                        quadMinPts = quadrantCount[quadY, quadX];
                    }
                }

           if (quadMinY == 0 && quadMinX == 0) 
                return ScottPlot.Alignment.LowerLeft;

            if (quadMinY == 1 && quadMinX == 0)
                return ScottPlot.Alignment.UpperLeft;

            if (quadMinY == 1 && quadMinX == 1)
                return ScottPlot.Alignment.UpperRight;

            return ScottPlot.Alignment.LowerRight;
        }

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
        /// <param name="legend_alignment">Alignment of the legend frame (see "ScottPlot.Alignment")</param>
        public static void PlotFunction(SeriesData data, string plotFileName, int plotWidth = 600, int plotHeight = 800, string xLabel = "X", string yLabel = "Y", string title = "Title",
            System.Int32 legendAlignment = -1)
        {
            var plt = new ScottPlot.Plot(plotWidth, plotHeight);
            plt.XLabel(xLabel);
            plt.YLabel(yLabel);
            plt.Legend(location:(legendAlignment >=0 ? (ScottPlot.Alignment)legendAlignment : EstimateLegendAlignment(data)));
            plt.XTicks(data.X, data.X.Select(x => x.ToString()).ToArray());

            foreach (var series in data.Series)
            {
                if (data.X.Length != series.Y.Length)
                    throw new Exception("Incompatible input data");
                plt.AddScatter(data.X, series.Y, label: series.Label, lineWidth: series.LineWidth);
            }

            plt.Title(title); plt.SaveFig(plotFileName);
        }
    }
}
