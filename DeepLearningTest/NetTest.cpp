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

#include "CppUnitTest.h"
#include <NeuralNet/Net.h>
#include <MnistDataUtils.h>
#include <filesystem>
#include <MsgPackUtils.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NetTest)
	{
		/// <summary>
		/// Scales intensities of the given images so that they are all between 0 and 1.0;
		/// </summary>
		static std::vector<DenseVector> scale_images(const std::vector<Image8Bit>& images, const Real& one_value = Real(1))
		{
			std::vector<DenseVector> result(images.begin(), images.end());

			const auto scale_factor = one_value / 256;

			for (auto& result_item : result)
				result_item *= scale_factor;

			return result;
		}

		/// <summary>
		/// Return collections of data and labels (in this exact order)
		/// </summary>
		static std::tuple<std::vector<DenseVector>, std::vector<DenseVector>> load_labeled_data(
			const std::filesystem::path& data_path, const std::filesystem::path& labels_path, const std::size_t expected_items_count, const Real& one_value = Real(1))
		{
			const auto images = MnistDataUtils::read_images(data_path, expected_items_count);
			const auto images_scaled = scale_images(images, one_value);
			const auto labels = MnistDataUtils::read_labels(labels_path, expected_items_count);

			return std::make_tuple(images_scaled, labels);
		}

		/// <summary>
		/// A general method to run MNIST-based training and evaluation
		/// </summary>
		/// <param name="cost_func_id">Id of the cost function we want to use for training.</param>
		/// <param name="learning_rate">The learning rate we want to use.</param>
		/// <param name="expected_min_percentage_test_set">Expected minimal percentage of correct answers of the test data after the training.
		/// Can take values from (0, 1).</param>
		static void RunMnistBasedTrainingTest(const CostFunctionId cost_func_id, const Real& learning_rate, const Real& expected_min_percentage_test_set, const bool run_long_test)
		{
			//Arrange
			const auto training_images_count = 60000;
			const auto [training_data, training_labels] = load_labeled_data(
				"TestData\\MNIST\\train-images.idx3-ubyte",
				"TestData\\MNIST\\train-labels.idx1-ubyte",
				training_images_count);

			const auto test_images_count = 10000;
			const auto [test_data, test_labels] = load_labeled_data(
				"TestData\\MNIST\\t10k-images.idx3-ubyte",
				"TestData\\MNIST\\t10k-labels.idx1-ubyte",
				test_images_count);

			auto net = Net({ 784, (run_long_test ? 100ull : 30ull), 10 });
			const auto batch_size = 10;
			const auto long_test = true;
			const auto epochs_count = run_long_test ? 30 : 5;

			//Act
			const auto costs_per_epoch = net.learn(training_data, training_labels, batch_size, epochs_count, learning_rate, cost_func_id);

			//Report some information about how fast we converge and what was the value of the cost function after each training epoch
			for (std::size_t cost_id = 0; cost_id < costs_per_epoch.size(); cost_id++)
				Logger::WriteMessage((std::string("Epoch : ") + std::to_string(cost_id) +
					std::string("; Cost : ") + std::to_string(costs_per_epoch[cost_id]) + "\n").c_str());

			//Assert
			Assert::IsTrue(costs_per_epoch.size() == epochs_count,
				L"Number of elements in the output collection should be equal to the number of epochs.");

			Real correct_unswers = 0;
			for (std::size_t test_item_id = 0; test_item_id < test_data.size(); test_item_id++)
			{
				const auto& test_item = test_data[test_item_id];
				const auto ref_answer = test_labels[test_item_id].max_element_id();
				const auto trial_label = net.act(test_item);
				const auto abs_sum = trial_label.sum();
				const auto trial_label_normalized = trial_label * (Real(1) / abs_sum);
				const auto trial_answer = trial_label_normalized.max_element_id();

				if (trial_answer == ref_answer && trial_label_normalized(trial_answer) > Real(0.5))
					correct_unswers += Real(1);
			}

			Logger::WriteMessage((std::string("Correct answers : ") + std::to_string(correct_unswers) + "\n").c_str());

			Assert::IsTrue(correct_unswers/ test_data.size() >= expected_min_percentage_test_set, L"Too low accuracy on the test set.");
		}

	public:
		TEST_METHOD(TrainingWithQuadraticCostTest)
		{
			const bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::SQUARED_ERROR, Real(3.0), long_test ? 0.97 : 0.94, long_test);
		}

		TEST_METHOD(TrainingWithCrossEntropyCostTest)
		{
			const bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, Real(0.5), long_test ? 0.97 : 0.94, long_test);
		}

		TEST_METHOD(NetSerializationTest)
		{
			//Arrange
			const auto in_dimension = 784;
			const auto out_dimension = 10;
			const auto net = Net({ in_dimension, 100, out_dimension });

			//Act
			const auto msg = MsgPack::pack(net);
			const auto net_unpacked = MsgPack::unpack<Net>(msg);

			//Assert
			const auto tests_samples_count = 100;
			for (int test_sample_id = 0; test_sample_id < tests_samples_count; test_sample_id++)
			{
				//take a random input sample
				const auto input_sample = DenseVector(in_dimension, -1, 1);
				const auto ref_output = net.act(input_sample);
				//Sanity check
				Assert::IsTrue(ref_output.max_abs() > 0 && input_sample.max_abs() > 0, L"Both input sample and reference output are expected to be non-zero.");

				const auto trial_output = net_unpacked.act(input_sample);

				Assert::IsTrue(ref_output == trial_output, L"Nets are not the same.");
			}
		}
	};
}
