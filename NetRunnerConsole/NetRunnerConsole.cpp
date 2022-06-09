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

#include <iostream>
#include <filesystem>
#include <sstream>
#include <tclap/CmdLine.h>
#include <defs.h>
#include <NeuralNet/Net.h>
#include "LearningRateReporter.h"
#include "MnistDataUtils.h"
#include "Math/CostFunction.h"
#include "Utilities.h"
#include <chrono>
#include <time.h>
#include <format>
#include <string>

using namespace DeepLearning;
using namespace NetRunnerConsole;

/// <summary>
/// Returns time-string formated as "day of week"-"month"-"date"-hh-mm-ss-year
/// </summary>
/// <param name="time_pt"></param>
std::string format_time(const std::chrono::system_clock::time_point& time_pt)
{
	std::time_t t = std::chrono::system_clock::to_time_t(time_pt);
	char str[100];
	ctime_s(str, sizeof str,  &t);
	const auto cnt = std::find(str, str + sizeof(str), '\0') - &str[0];
	auto result = Utils::remove_leading_trailing_extra_spaces(std::string(str, cnt-1));
	std::replace_if(result.begin(), result.end(),[](const auto x) {
			return x == ' ' || x == ':';
		}, '-');

	return result;
}

/// <summary>
/// Returns formated duration between the start and stop time points
/// </summary>
std::string get_elapsed_time_formated(const std::chrono::system_clock::time_point& start_pt, const std::chrono::system_clock::time_point& stop_pt)
{
	const auto epoch_time_sec = std::chrono::duration_cast<std::chrono::seconds>(stop_pt - start_pt);
	std::string result = std::format("{:%T}", epoch_time_sec);
	return result;
}

/// <summary>
/// Return string with formated elapsed time and updates given time-point with new measurement
/// </summary>
/// <param name="start_pt">Start time point</param>
/// <returns></returns>
std::string get_elapsed_time_formated(std::chrono::system_clock::time_point& start_pt)
{
	const auto stop_pt = std::chrono::system_clock::now();
	const auto result = get_elapsed_time_formated(start_pt, stop_pt);
	start_pt = stop_pt;
	return result;
}

int main(int argc, char** argv)
{

	TCLAP::CmdLine cmd("Network teaching engine", ' ', "1.0");

	auto script_arg = TCLAP::ValueArg<std::string>("s", "script", "Full file path to a script to instantiate a neural network", true, "", "string");
	cmd.add(script_arg);

	auto rate_arg = TCLAP::ValueArg<Real>("r", "rate", "Learning rate for the network", false, 0.1, "double");
	cmd.add(rate_arg);

	auto minibatch_arg = TCLAP::ValueArg<int>("m", "minibatch", "Number of items in a mini-batch", false, 10, "integer");
	cmd.add(minibatch_arg);

	auto epoch_arg = TCLAP::ValueArg<int>("e", "epochs", "Number of epochs to do", false, 3, "integer");
	cmd.add(epoch_arg);

	auto iteration_arg = TCLAP::ValueArg<int>("i", "iterations", "Number teaching iterations", false, 2, "integer");
	cmd.add(iteration_arg);

	auto reg_factor_arg = TCLAP::ValueArg<Real>("l", "lambda", "L2 regularization factor", false, 0.0, "double");
	cmd.add(reg_factor_arg);

	cmd.parse(argc, argv);

	const auto batch_size = minibatch_arg.getValue();
	const auto epochs_count = epoch_arg.getValue();
	const auto learning_rate = rate_arg.getValue();
	const auto reg_factor = reg_factor_arg.getValue();

	const std::filesystem::path script_path = script_arg.getValue();
	Net net_original;
	try
	{
		net_original = Net::load_script(script_path);
	}
	catch (const std::exception& e)
	{
		std::cout << "Failed to parse net script with exception: " << e.what() << std::endl;
		return 0;
	}

	const auto summary = std::string("Mini-batch size : ") + std::to_string(batch_size) + "\n" +
		"Epochs count : " + std::to_string(epochs_count) + "\n" +
		"Learning rate : " + Utils::to_string(learning_rate) + "\n" +
		"Regularization factor : " + Utils::to_string(reg_factor) + "\n" + net_original.to_string() + "\n";

	std::cout << summary;

	const auto training_images_count = 60000;
	const auto training_data_tuple = MnistDataUtils::load_labeled_data(
		"TestData\\MNIST\\train-images.idx3-ubyte",
		"TestData\\MNIST\\train-labels.idx1-ubyte",
		training_images_count, /*flatten images*/false);

	const auto& training_data = std::get<0>(training_data_tuple);
	const auto& training_labels = std::get<1>(training_data_tuple);

	const auto test_images_count = 10000;
	const auto test_data_tuple = MnistDataUtils::load_labeled_data(
		"TestData\\MNIST\\t10k-images.idx3-ubyte",
		"TestData\\MNIST\\t10k-labels.idx1-ubyte",
		test_images_count, /*flatten images*/false);

	const auto& test_data = std::get<0>(test_data_tuple);
	const auto& test_labels = std::get<1>(test_data_tuple);

	const auto directory = script_path.parent_path();

	const auto cost_func_id = CostFunctionId::CROSS_ENTROPY;

	LRReporter reporter(summary, " ");

	const auto start = std::chrono::system_clock::now();
	auto epoch_start = start;
	std::cout << "Started at: " << format_time(start) << std::endl;
	for (auto iter_id = 1; iter_id <= iteration_arg.getValue(); iter_id++)
	{
		auto net_to_train = Net(net_original.to_script());

		reporter.new_training();

		const auto evaluation_action = [&](const auto epoch_id, const auto scaled_l2_reg_factor)
		{
			const auto correct_answers_test_data = net_to_train.count_correct_answers(test_data, test_labels) * Real(1) / test_data.size();
			const auto correct_answers_training_data = net_to_train.count_correct_answers(training_data, training_labels) * Real(1) / training_data.size();
			const auto cost_function = net_to_train.evaluate_cost_function(training_data, training_labels, cost_func_id, scaled_l2_reg_factor);
			reporter.add_data(correct_answers_test_data, correct_answers_training_data, cost_function);
			const auto elapsed_time_str = get_elapsed_time_formated(epoch_start);
			std::cout << "Iteration : " << iter_id << "; Epoch : " << epoch_id << "; success rate : "
				<< correct_answers_test_data << " %; time: " << elapsed_time_str << std::endl;
		};

		net_to_train.learn(training_data, training_labels, batch_size, epochs_count,
			learning_rate, cost_func_id, reg_factor, evaluation_action);
	}

	const auto stop = std::chrono::system_clock::now();
	std::cout << "Finished at: " << format_time(stop) << std::endl;
	std::cout << "Total execution time : " << get_elapsed_time_formated(start, stop) << std::endl;

	//Save report
	const auto start_time_str = format_time(start);
	const auto end_time_str = format_time(stop);
	const auto report_base_name = script_path.stem().string() + "_" + start_time_str + "_" + end_time_str + "_report.txt";
	const auto report_full_path = directory / report_base_name;
	reporter.save_report(report_full_path);
}
