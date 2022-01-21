#include "CppUnitTest.h"
#include <NeuralNet/Net.h>
#include <MnistDataUtils.h>
#include <filesystem>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NetTest)
	{
		/// <summary>
		/// Scales intensities of the given images so that they are all between 0 and 1.0;
		/// </summary>
		std::vector<DenseVector> scale_images(const std::vector<Image8Bit>& images)
		{
			std::vector<DenseVector> result(images.begin(), images.end());

			const auto scale_factor = Real(1) / 256;

			for (auto& result_item : result)
				result_item *= scale_factor;

			return result;
		}

		/// <summary>
		/// Return collections of data and labels (in this exact order)
		/// </summary>
		std::tuple<std::vector<DenseVector>, std::vector<DenseVector>> load_labeled_data(
			const std::filesystem::path& data_path, const std::filesystem::path& labels_path, const std::size_t expected_items_count)
		{
			const auto images = MnistDataUtils::read_images(data_path, expected_items_count);
			const auto images_scaled = scale_images(images);
			const auto labels = MnistDataUtils::read_labels(labels_path, expected_items_count);

			return std::make_tuple(images_scaled, labels);
		}

	public:
		TEST_METHOD(LearningTest)
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

			auto net = Net({784, 100, 10});
			const auto batch_size = 10;
			const auto long_test = false;
			const auto epochs_count = long_test ? 30 : 3;
			const auto learning_rate = Real(3.0);

			//Act
			const auto costs_per_epoch =  net.learn(training_data, training_labels, batch_size, epochs_count, learning_rate, CostFunctionId::SQUARED_ERROR);

			for (std::size_t cost_id = 0; cost_id < costs_per_epoch.size(); cost_id++)
			{
				Logger::WriteMessage((std::string("Epoch : ") + std::to_string(cost_id) + std::string("; Cost : ") + std::to_string(costs_per_epoch[cost_id]) + "\n").c_str());
			}

			//Assert
			Assert::IsTrue(costs_per_epoch.size() == epochs_count,
				L"Number of elements in the output collection should be equal to the number of epochs.");

			int correct_unswers = 0;
			for (std::size_t test_item_id = 0; test_item_id < test_data.size(); test_item_id++)
			{
				const auto& test_item = test_data[test_item_id];
				const auto ref_answer = test_labels[test_item_id].max_element_id();
				const auto trial_label = net.act(test_item);
				const auto abs_sum = trial_label.sum();
				const auto trial_label_normalized = trial_label * (Real(1) / abs_sum);
				const auto trial_answer = trial_label_normalized.max_element_id();

				if (trial_answer == ref_answer && trial_label_normalized(trial_answer) > Real(0.5))
					correct_unswers++;
			}

			Logger::WriteMessage((std::string("Correct answers : ") + std::to_string(correct_unswers) + "\n").c_str());

			Assert::IsTrue(correct_unswers >= (long_test ? 9700 : 9500), L"Too low accuracy on the test set.");
		}
	};
}
