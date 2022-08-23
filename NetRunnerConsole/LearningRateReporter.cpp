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

#include "LearningRateReporter.h"
#include <fstream>
#include <exception>
#include <Utilities.h>
#include <numeric>
#include "../ManagedToUnmanagedBridge/ManagedToUnmanagedBridge.h"

namespace NetRunnerConsole
{
	void LRReporter::new_training()
	{
		_data.push_back(TrainingReport());
	}

	ScottPlotBridge::SeriesData data;

	void LRReporter::plot_single_training_parameter(const std::filesystem::path& base_file_path, const std::string& parameter_tag, const TrainingReport& average_report, const std::size_t& param_id) const
	{
		ScottPlotBridge::SeriesData data;
		const auto max_series_size = calc_max_epoch_size();
		data.x.resize(max_series_size);
		std::iota(data.x.begin(), data.x.end(), 1);
		data.series.resize(_data.size() + 1);

		for (auto training_iter_id = 0; training_iter_id < _data.size(); training_iter_id++)
		{
			const auto& epoh_data = _data[training_iter_id];
			auto& current_series = data.series[training_iter_id];
			current_series.label = "Training " + std::to_string(training_iter_id);
			current_series.y.resize(max_series_size, 0);

			for (auto epoch_id = 0ull; epoch_id < epoh_data.size(); epoch_id++)
				current_series.y[epoch_id] = epoh_data[epoch_id][param_id];
		}

		auto& averaged_series = data.series[_data.size()];
		averaged_series.label = "Averaged";
		averaged_series.y.resize(max_series_size, 0);
		averaged_series.line_width = 3;
		for (auto epoch_id = 0ull; epoch_id < average_report.size(); epoch_id++)
			averaged_series.y[epoch_id] = average_report[epoch_id][param_id];

		auto base_path_writable = base_file_path;
		ScottPlotBridge::PlotFunction(data, base_path_writable.replace_extension(parameter_tag + ".png"), 2000, 800, "Epoch", parameter_tag, parameter_tag);
	}

	/// <summary>
	/// Functionality to calculate min and max values while iterating series of numeric point values
	/// </summary>
	struct MinMax
	{
		Real min = std::numeric_limits<Real>::max();
		Real max = -std::numeric_limits<Real>::max();

		/// <summary>
		/// Adds value to the structure
		/// </summary>
		void Add(const Real val)
		{
			max = std::max(max, val);
			min = std::min(min, val);
		}
	};

	void LRReporter::write_single_training_parameter(std::ofstream& file,
		const TrainingReport& average_report, const std::size_t& param_id) const
	{
		MinMax min_max_total;
		for (auto training_iter_id = 0; training_iter_id < _data.size(); training_iter_id++)
		{
			const auto& epoch_data = _data[training_iter_id];
			for (auto epoch_id = 0ull; epoch_id < epoch_data.size(); epoch_id++)
			{
				min_max_total.Add(epoch_data[epoch_id][param_id]);
				file << epoch_data[epoch_id][param_id] << _delimiter;
			}
			file << std::endl;
		}
		file << "Min: " << min_max_total.min << _delimiter << " Max: " << min_max_total.max << _delimiter << std::endl;

		file << "Average:" << std::endl;

		MinMax min_max;
		for (auto epoch_id = 0ull; epoch_id < average_report.size(); epoch_id++)
		{
			min_max.Add(average_report[epoch_id][param_id]);
			file << average_report[epoch_id][param_id] << _delimiter;
		}
		file << " Min: " << min_max.min << _delimiter << " Max: " << min_max.max << _delimiter << std::endl;
	}

	void LRReporter::save_report(const std::filesystem::path& report_name) const
	{
		std::ofstream file(report_name);

		if (!file.is_open())
			throw std::exception("Can't create file");

		file << _description << std::endl;

		const auto average_report = report_average();

		for (auto param_id = 0ull; param_id < ReportItem::params_count(); param_id++)
		{
			const auto parameter_description = ReportItem::get_param_description(param_id);
			file << parameter_description << ":" << std::endl;
			write_single_training_parameter(file, average_report, param_id);
			file << std::endl;

			plot_single_training_parameter(report_name, parameter_description, average_report, param_id);
		}
	}

	void LRReporter::add_data(const Real& test_data_success_rate, const Real& training_data_success_rate, const Real& cost_function)
	{
		if (_data.empty())
			new_training();

		_data.rbegin()->emplace_back(test_data_success_rate, training_data_success_rate, cost_function);
	}

	Vector3d<Real> LRReporter::ReportItem::to_vector() const
	{
		return { test_data_success_rate, training_data_success_rate, cost_function };
	}

	LRReporter::ReportItem::ReportItem(const Vector3d<Real>& source)
	{
		test_data_success_rate = source.x;
		training_data_success_rate = source.y;
		cost_function = source.z;
	}

	LRReporter::ReportItem::ReportItem(const Real& test_data_success_rate_, const Real& training_data_success_rate_, const Real& cost_function_)
	{
		test_data_success_rate = test_data_success_rate_;
		training_data_success_rate = training_data_success_rate_;
		cost_function = cost_function_;
	}

	std::size_t LRReporter::calc_max_epoch_size() const
	{
		return std::max_element(_data.begin(), _data.end(),
			[](const auto& x, const auto& y) { return x.size() < y.size(); })->size();
	}

	LRReporter::TrainingReport LRReporter::report_average() const
	{
		TrainingReport result{};

		if (_data.empty())
			return result;

		const auto elements_cnt = calc_max_epoch_size();

		for (auto elem_id = 0ull; elem_id < elements_cnt; elem_id++)
		{
			Vector3d<Real> average_vec{};
			int items_cnt = 0;
			for (auto training_id = 0ull; training_id < _data.size(); training_id++)
			{
				if (_data[training_id].size() > elem_id)
				{
					average_vec += _data[training_id][elem_id].to_vector();
					items_cnt++;
				}
			}
			average_vec *= (Real(1) / items_cnt);
			result.emplace_back(average_vec);
		}

		return result;
	}

	const Real& LRReporter::ReportItem::operator [](const std::size_t& param_id) const
	{
		switch (param_id)
		{
		case 0: return test_data_success_rate;
		case 1: return training_data_success_rate;
		case 2: return cost_function;
		default:
			throw std::exception("Impossible parameter identifier");
		}
	}

	std::string LRReporter::ReportItem::get_param_description(const std::size_t& param_id)
	{
		switch (param_id)
		{
		case 0: return "Success rate on the test data set";
		case 1: return "Success rate on the training data set";
		case 2: return "Cost function value";
		default:
			throw std::exception("Impossible parameter identifier");
		}
	}


}