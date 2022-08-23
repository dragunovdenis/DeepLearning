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

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <defs.h>
#include <Math/LinAlg3d.h>

using namespace DeepLearning;

namespace NetRunnerConsole
{
	/// <summary>
	///"Learning rate reporter" functionality for tracking data about learning
	/// rates of different network architectures (on different learning data)
	/// </summary>
	class LRReporter
	{
		/// <summary>
		/// Data structure to hold a minimal piece of report information
		/// </summary>
		struct ReportItem
		{
			/// <summary>
			/// Success rate of the net evaluated at the test data set
			/// </summary>
			Real test_data_success_rate{};

			/// <summary>
			/// Success rate on the training data set
			/// </summary>
			Real training_data_success_rate{};

			/// <summary>
			/// Value of the cost function used in the learning process (evaluated at the training data set)
			/// </summary>
			Real cost_function{};

			/// <summary>
			/// Conversion to 3d vector (to facilitate arithmetic operations with the item)
			/// </summary>
			Vector3d<Real> to_vector() const;

			/// <summary>
			/// Default constructor
			/// </summary>
			ReportItem() = default;

			/// <summary>
			/// Constructor
			/// </summary>
			ReportItem(const Real& test_data_success_rate_, const Real& training_data_success_rate_, const Real& cost_function_);

			/// <summary>
			/// Instantiation from a 3d vector
			/// </summary>
			ReportItem(const Vector3d<Real>& source);

			/// <summary>
			/// Returns description of the parameter item with the given identifier
			/// </summary>
			static std::string get_param_description(const std::size_t& param_id) ;

			/// <summary>
			/// Returns number of parameters in the item (allows to figure out the valid range for the item identifiers)
			/// </summary>
			static std::size_t params_count() { return sizeof(ReportItem)/sizeof(Real); };

			/// <summary>
			/// Read-only subscript operator
			/// </summary>
			const Real& operator [](const std::size_t& param_id) const;
		};

		/// <summary>
		/// Data structure to hold report date of a single training process
		/// </summary>
		using TrainingReport = std::vector<ReportItem>;

		const std::string _description{};

		/// <summary>
		/// Delimiter to be used to separate items in a single row of the text report 
		/// </summary>
		const std::string _delimiter;

		std::vector<TrainingReport> _data{};

		/// <summary>
		/// Writes report for the given parameter (represented with the given parameter identifier)
		/// into the given file stream
		/// </summary>
		void write_single_training_parameter(std::ofstream& file, const TrainingReport& average_report, const std::size_t& param_id) const;

		/// <summary>
		/// Plots data for the given parameter (represented with the given parameter identifier)
		/// and saves it to the given folder in "png" format
		/// </summary>
		void plot_single_training_parameter(const std::filesystem::path& base_file_path, const std::string& parameter_tag, const TrainingReport& average_report, const std::size_t& param_id) const;

		/// <summary>
		/// Returns number of epochs in the longest training series
		/// </summary>
		std::size_t calc_max_epoch_size() const;
	public:
		/// <summary>
		/// Constructor
		/// </summary>
		LRReporter(const std::string& description, const std::string& delim = " ") : _description(description), _delimiter(delim){}

		/// <summary>
		/// Method to add a single piece of report data
		/// </summary>
		void add_data(const Real& test_data_success_rate, const Real& training_data_success_rate, const Real& cost_function);

		/// <summary>
		/// Initiates beginning of a new training process
		/// </summary>
		void new_training();

		/// <summary>
		/// Saves report to the file with the given name on the disk in text format
		/// </summary>
		void save_report(const std::filesystem::path& report_name) const;

		/// <summary>
		/// Returns "averaged" report
		/// </summary>
		TrainingReport report_average() const;
	};

}
