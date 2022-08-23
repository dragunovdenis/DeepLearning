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

#include "ManagedToUnmanagedBridge.h"

using namespace System;

namespace ScottPlotBridge
{
    /// <summary>
    /// Converts given "native" vector to a managed array
    /// </summary>
    template <class T>
    array<T>^ vector_to_managed_array(const std::vector<T>& vector)
    {
        const auto elements_count = static_cast<int>(vector.size());
        auto result = gcnew array<T>(elements_count);

        for (auto i = 0; i < elements_count; ++i)
            result[i] = vector[i];

        return result;
    }

    /// <summary>
    /// Converts unmanaged "series data" to its managed version
    /// </summary>
    ScottPlotWrapper::SeriesData^ to_managed_series_data(const SeriesData& data)
    {
        auto result = gcnew ScottPlotWrapper::SeriesData();
        const auto series_count = static_cast<int>(data.series.size());

        result->X = vector_to_managed_array(data.x);
        result->Series = gcnew array<ScottPlotWrapper::YSeries^>(series_count);

        for (auto series_id = 0; series_id < series_count; ++series_id)
        {
            const auto y_series_managed = gcnew ScottPlotWrapper::YSeries();
            y_series_managed->Y = vector_to_managed_array(data.series[series_id].y);
            y_series_managed->Label = gcnew String(data.series[series_id].label.c_str());
            y_series_managed->LineWidth = data.series[series_id].line_width;
            result->Series[series_id] = y_series_managed;
        }

        return result;
    }

    void PlotFunction(const SeriesData& data, const std::filesystem::path& plotFileName,
        const int plot_width, const int plot_height, const std::string& x_label,
        const std::string& y_label, const std::string& title, const LegendAlignment legend_alignment)
    {
        const auto series_data_managed = to_managed_series_data(data);
        const auto path_string_managed = gcnew String(plotFileName.c_str());
        const auto x_label_managed = gcnew String(x_label.c_str());
        const auto y_label_managed = gcnew String(y_label.c_str());
        const auto title_managed = gcnew String(title.c_str());
        ScottPlotWrapper::ScottPlotter::PlotFunction(series_data_managed, path_string_managed,
            plot_width, plot_height, x_label_managed, y_label_managed, title_managed, (int)legend_alignment);
    }
}


