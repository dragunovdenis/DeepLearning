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
#include <tclap/CmdLine.h>
#include <defs.h>
#include <NeuralNet/Net.h>
#include "LearningRateReporter.h"
#include "MnistDataUtils.h"
#include "Math/CostFunction.h"
#include "Utilities.h"
#include <chrono>
#include <string>

using namespace DeepLearning;
using namespace NetRunnerConsole;

int main(int argc, char** argv)
{

	TCLAP::CmdLine cmd("Network teaching engine", ' ', "1.0");

	auto script_arg = TCLAP::ValueArg<std::string>("s", "script", "Full file path to a script to instantiate a neural network", true, "", "string");
	cmd.add(script_arg);

	auto rate_arg = TCLAP::ValueArg<Real>("r", "rate", "Learning rate for the network", false, static_cast<Real>(0.1), "double");
	cmd.add(rate_arg);

	auto minibatch_arg = TCLAP::ValueArg<int>("m", "minibatch", "Number of items in a mini-batch", false, 10, "integer");
	cmd.add(minibatch_arg);

	auto epoch_arg = TCLAP::ValueArg<int>("e", "epochs", "Number of epochs to do", false, 3, "integer");
	cmd.add(epoch_arg);

	auto iteration_arg = TCLAP::ValueArg<int>("i", "iterations", "Number teaching iterations", false, 2, "integer");
	cmd.add(iteration_arg);

	auto reg_factor_arg = TCLAP::ValueArg<Real>("l", "lambda", "L2 regularization factor", false, 0.0, "double");
	cmd.add(reg_factor_arg);

	auto cost_func_arg = TCLAP::ValueArg<std::string>("c", "cost", "Cost function used in the training process", false, "CROSS_ENTROPY", "string");
	cmd.add(cost_func_arg);

	auto transform_arg = TCLAP::ValueArg<std::string>("t", "transform", 
		"Transformations to construct extra training data {`rotation angle around image center`, `translation X`, `translation Y`}", false, "", "string");
	cmd.add(transform_arg);

	cmd.parse(argc, argv);

	const auto batch_size = minibatch_arg.getValue();
	const auto epochs_count = epoch_arg.getValue();
	const auto learning_rate = rate_arg.getValue();
	const auto reg_factor = reg_factor_arg.getValue();
	const auto cost_func_id = parse_cost_type(cost_func_arg.getValue());
	const auto transformations = Utils::extract_vectors<Vector3d<Real>>(transform_arg.getValue());

	if (cost_func_id == CostFunctionId::UNKNOWN)
	{
		std::cout << "Invalid type of the cost function!!! Please check spelling." << std::endl;
		return 0;
	}

	const std::filesystem::path script_path = script_arg.getValue();
	Net net_to_train;
	try
	{
		net_to_train.try_load_from_script_file(script_path);
	}
	catch (const std::exception& e)
	{
		std::cout << "Failed to parse net script with exception: " << e.what() << std::endl;
		return 0;
	}

	const auto summary = std::string("Mini-batch size : ") + std::to_string(batch_size) + "\n" +
		"Epochs count : " + std::to_string(epochs_count) + "\n" +
		"Learning rate : " + Utils::to_string(learning_rate) + "\n" +
		"Regularization factor : " + Utils::to_string(reg_factor) + "\n" +
		"Cost function : " + to_string(cost_func_id) + "\n" +
		"Transformations :\n" + Utils::to_string(transformations) + "\n" +
		"Net architecture: " + "\n" + net_to_train.to_string() + "\n";

	std::cout << summary;

	constexpr auto training_images_count = 60000;
	const auto training_data_tuple = MnistDataUtils::load_labeled_data(
		"TestData\\MNIST\\train-images.idx3-ubyte",
		"TestData\\MNIST\\train-labels.idx1-ubyte",
		training_images_count, /*max value*/ Real(1), transformations);

	const auto& training_data = std::get<0>(training_data_tuple);
	const auto& training_labels = std::get<1>(training_data_tuple);

	constexpr auto test_images_count = 10000;
	const auto test_data_tuple = MnistDataUtils::load_labeled_data(
		"TestData\\MNIST\\t10k-images.idx3-ubyte",
		"TestData\\MNIST\\t10k-labels.idx1-ubyte",
		test_images_count);

	const auto& test_data = std::get<0>(test_data_tuple);
	const auto& test_labels = std::get<1>(test_data_tuple);

	const auto directory = script_path.parent_path();

	LRReporter reporter(summary, "; ");

	const auto start = std::chrono::system_clock::now();
	auto epoch_start = start;
	std::cout << "Started at: " << Utils::format_date_time(start) << std::endl;
	for (auto iter_id = 1; iter_id <= iteration_arg.getValue(); iter_id++)
	{
		net_to_train.try_load_from_script_file(script_path); //re-instantiate the net (to start from scratch, so to speak)
		reporter.new_training();

		const auto evaluation_action = [&](const auto epoch_id, const auto scaled_l2_reg_factor)
		{
			const auto correct_answers_test_data = net_to_train.count_correct_answers(test_data, test_labels) * Real(1) / test_data.size();
			const auto cost_and_answers = net_to_train.evaluate_cost_function_and_correct_answers(training_data, training_labels, cost_func_id, scaled_l2_reg_factor);
			reporter.add_data(correct_answers_test_data, cost_and_answers.CorrectAnswers, cost_and_answers.Cost);
			const auto elapsed_time_str = Utils::get_elapsed_time_formatted(epoch_start);
			std::cout << "Iteration : " << iter_id << "; Epoch : " << epoch_id << "; success rate : "
				<< correct_answers_test_data << " %; time: " << elapsed_time_str << std::endl;
		};

		net_to_train.learn(training_data, training_labels, batch_size, epochs_count,
			learning_rate, cost_func_id, reg_factor, evaluation_action);
	}

	const auto stop = std::chrono::system_clock::now();
	std::cout << "Finished at: " << Utils::format_date_time(stop) << std::endl;
	std::cout << "Total execution time : " << Utils::get_elapsed_time_formatted(start, stop) << std::endl;

	//Save report
	const auto start_time_str = Utils::format_date_time(start);
	const auto end_time_str = Utils::format_date_time(stop);
	const auto report_base_name = script_path.stem().string() + "_" + start_time_str + "_" + end_time_str + "_report.txt";
	const auto report_full_path = directory / report_base_name;
	reporter.save_report(report_full_path);

	return 0;
}
