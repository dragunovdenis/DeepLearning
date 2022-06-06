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
#include <chrono>

using namespace DeepLearning;
using namespace NetRunnerConsole;

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
	auto [training_data, training_labels] = MnistDataUtils::load_labeled_data(
		"TestData\\MNIST\\train-images.idx3-ubyte",
		"TestData\\MNIST\\train-labels.idx1-ubyte",
		training_images_count, /*flatten images*/false);

	const auto test_images_count = 10000;
	const auto [test_data, test_labels] = MnistDataUtils::load_labeled_data(
		"TestData\\MNIST\\t10k-images.idx3-ubyte",
		"TestData\\MNIST\\t10k-labels.idx1-ubyte",
		test_images_count, /*flatten images*/false);

	const auto directory = script_path.parent_path();

	const auto cost_func_id = CostFunctionId::CROSS_ENTROPY;

	LRReporter reporter(summary, " ");

	const auto start = std::chrono::steady_clock::now();
	auto epoch_start = start;
	for (auto iter_id = 1; iter_id < iteration_arg.getValue(); iter_id++)
	{
		auto net_to_train = Net(net_original.to_script());

		reporter.new_training();

		const auto evaluation_action = [&](const auto epoch_id, const auto scaled_l2_reg_factor)
		{
			const auto correct_answers_test_data = net_to_train.count_correct_answers(test_data, test_labels) * Real(1) / test_data.size();
			const auto correct_answers_training_data = net_to_train.count_correct_answers(training_data, training_labels) * Real(1) / training_data.size();
			const auto cost_function = net_to_train.evaluate_cost_function(training_data, training_labels, cost_func_id, scaled_l2_reg_factor);
			reporter.add_data(correct_answers_test_data, correct_answers_training_data, cost_function);
			const auto epoch_stop = std::chrono::steady_clock::now();
			const auto epoch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_stop - epoch_start).count();
			epoch_start = epoch_stop;
			std::cout << "Iteration : " << iter_id << "; Epoch : " << epoch_id << "; success rate : "
				<< correct_answers_test_data << " %; time: " << epoch_time_ms << " ms." << std::endl;
		};

		net_to_train.learn(training_data, training_labels, batch_size, epochs_count,
			learning_rate, cost_func_id, reg_factor, evaluation_action);
	}
	const auto end = std::chrono::steady_clock::now();
	std::cout << "Execution time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;

	const auto report_base_name = script_path.stem().string() + "_report.txt";
	const auto report_full_path = directory / report_base_name;
	reporter.save_report(report_full_path);
}
