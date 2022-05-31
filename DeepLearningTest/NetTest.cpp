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
#include <NeuralNet/CLayer.h>
#include <NeuralNet/PLayer.h>
#include <MnistDataUtils.h>
#include <filesystem>
#include <MsgPackUtils.h>
#include <fstream>
#include <chrono>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NetTest)
	{
		/// <summary>
		/// Scales intensities of the given images so that they are all between 0 and 1.0;
		/// </summary>
		static std::vector<Tensor> scale_images(const std::vector<Image8Bit>& images, const bool flatten_images = true, const Real& max_value = Real(1))
		{
			if (images.empty())
				return std::vector<Tensor>();

			std::vector<Tensor> result(images.begin(), images.end());

			const auto image_size = result.begin()->size();
			if (!std::all_of(result.begin(), result.end(), [=](const auto& im) { return im.size() == image_size; }))
				throw std::exception("Images are supposed to be of same size");

			const auto scale_factor = max_value / 256;

			for (auto& result_item : result)
				result_item *= scale_factor;

			if (!flatten_images)
			{
				const auto shape = Index3d{ 1, images.begin()->height(), images.begin()->width() };
				for (auto& result_item : result)
					result_item.reshape(shape);
			}

			return result;
		}

		/// <summary>
		/// Return collections of data and labels (in this exact order)
		/// </summary>
		static std::tuple<std::vector<Tensor>, std::vector<Tensor>> load_labeled_data(
			const std::filesystem::path& data_path, const std::filesystem::path& labels_path,
			const std::size_t expected_items_count, const bool flatten_images = true, const Real& max_value = Real(1))
		{
			const auto images = MnistDataUtils::read_images(data_path, expected_items_count);
			const auto images_scaled = scale_images(images, flatten_images, max_value);
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
		static Real RunMnistBasedTrainingTest(const CostFunctionId cost_func_id, const Real& learning_rate,
			const Real& expected_min_percentage_test_set, const bool run_long_test, const Real& lambda = Real(0),
			const std::vector<ActivationFunctionId>& activ_func_ids = std::vector<ActivationFunctionId>())
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

			auto net = Net({ 784, (run_long_test ? 100ull : 30ull), 10 }, activ_func_ids);
			const auto batch_size = 10;
			const auto epochs_count = run_long_test ? 30 : 6;

			const auto evaluation_action = [&](const auto epoch_id)
			{
				const auto correct_unswers = net.count_correct_answers(test_data, test_labels);
				Logger::WriteMessage((std::string("Epoch: ") + std::to_string(epoch_id) +  ". Correct answers : " + std::to_string(correct_unswers) + "\n").c_str());
			};

			//Act
			auto start = std::chrono::steady_clock::now();
			net.learn(training_data, training_labels, batch_size, epochs_count,
				learning_rate, cost_func_id, lambda, evaluation_action);
			auto end = std::chrono::steady_clock::now();
			Logger::WriteMessage(("Learning time : " +
				std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) + " ms.").c_str());

			//Assert
			const auto correct_unswers = net.count_correct_answers(test_data, test_labels);
			const auto validation_result = correct_unswers * (Real(1)) / test_data.size();
			Assert::IsTrue(validation_result >= expected_min_percentage_test_set, L"Too low accuracy on the test set.");
			Logger::WriteMessage("\n");
			return validation_result;
		}

		/// <summary>
		/// A general method to run MNIST-based training and evaluation for convolutional neural network
		/// </summary>
		/// <param name="cost_func_id">Id of the cost function we want to use for training.</param>
		/// <param name="learning_rate">The learning rate we want to use.</param>
		/// <param name="expected_min_percentage_test_set">Expected minimal percentage of correct answers of the test data after the training.
		/// Can take values from (0, 1).</param>
		static Real RunMnistBasedConvolutionNetTrainingTest(const CostFunctionId cost_func_id, const Real& learning_rate,
			const Real& expected_min_percentage_test_set, const bool run_long_test, const Real& lambda = Real(0))
		{
			//Arrange
			const auto training_images_count = 60000;
			auto [training_data, training_labels] = load_labeled_data(
				"TestData\\MNIST\\train-images.idx3-ubyte",
				"TestData\\MNIST\\train-labels.idx1-ubyte",
				training_images_count, /*flatten images*/false);

			const auto test_images_count = 10000;
			const auto [test_data, test_labels] = load_labeled_data(
				"TestData\\MNIST\\t10k-images.idx3-ubyte",
				"TestData\\MNIST\\t10k-labels.idx1-ubyte",
				test_images_count, /*flatten images*/false);

			auto net = Net();
			const auto in_data_size = training_data.begin()->size_3d();
			const auto out_size = training_labels.begin()->size_3d().coord_prod();
			auto size_in_next = in_data_size;
			size_in_next = net.append_layer<CLayer>(size_in_next, Index2d{ 5 }, run_long_test ? 20 : 5, ActivationFunctionId::RELU);
			size_in_next = net.append_layer<PLayer>(size_in_next, Index2d{ 2 }, PoolTypeId::MAX);
			size_in_next = net.append_layer<CLayer>(size_in_next, Index2d{ 5 }, run_long_test ? 40 : 10, ActivationFunctionId::RELU);
			size_in_next = net.append_layer<PLayer>(size_in_next, Index2d{ 2 }, PoolTypeId::MAX);
			size_in_next = net.append_layer<NLayer>(size_in_next.coord_prod(), 100, ActivationFunctionId::RELU, Real(-1), Real(1), true);
			size_in_next = net.append_layer<NLayer>(size_in_next.coord_prod(), out_size, ActivationFunctionId::SOFTMAX, Real(-1), Real(1), true);

			Assert::IsTrue(out_size == size_in_next.coord_prod(), L"Unexpected size of the net output");

			const auto batch_size = 10;
			const auto epochs_count = run_long_test ? 30 : 5;

			std::chrono::steady_clock::time_point start;
			std::chrono::steady_clock::time_point epoch_start;

			const auto evaluation_action = [&](const auto epoch_id)
			{
				const auto correct_unswers = net.count_correct_answers(test_data, test_labels);
				Logger::WriteMessage((std::string("Epoch: ") + std::to_string(epoch_id) + ". Correct answers : " + std::to_string(correct_unswers)).c_str());
				auto epoch_end = std::chrono::steady_clock::now();
				Logger::WriteMessage((" Epoch time : " +
					std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start).count()) + " ms.\n").c_str());
				epoch_start = epoch_end;
			};

			//Act
			start = std::chrono::steady_clock::now();
			epoch_start = start;
			net.learn(training_data, training_labels, batch_size, epochs_count,
				learning_rate, cost_func_id, lambda, evaluation_action);
			auto end = std::chrono::steady_clock::now();
			Logger::WriteMessage(("Learning time : " +
				std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) + " ms.").c_str());

			//Assert
			const auto correct_unswers = net.count_correct_answers(test_data, test_labels);
			const auto validation_result = correct_unswers * (Real(1)) / test_data.size();
			Assert::IsTrue(validation_result >= expected_min_percentage_test_set, L"Too low accuracy on the test set.");
			Logger::WriteMessage("\n");
			return validation_result;
		}


	public:

		TEST_METHOD(TrainingConvolutionNetWithCrossEntropyCostTest)
		{
			const bool long_test = false;
			RunMnistBasedConvolutionNetTrainingTest(CostFunctionId::CROSS_ENTROPY, Real(0.03),
				long_test ? Real(0.991) : Real(0.98), long_test, Real(3));
		}

		TEST_METHOD(TrainingWithQuadraticCostTest)
		{
			const bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::SQUARED_ERROR, Real(1.0), long_test ? Real(0.976) : Real(0.95), long_test);
		}

		TEST_METHOD(TrainingWithQuadraticCostRegularizedTest)
		{
			const bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::SQUARED_ERROR, Real(1.0), long_test ? Real(0.976) : Real(0.95), long_test, Real(1.0));
		}

		TEST_METHOD(TrainingWithCrossEntropyCostTest)
		{
			const bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, Real(0.5), long_test ? Real(0.977) : Real(0.95), long_test);
		}

		TEST_METHOD(TrainingWithCrossEntropyCostRegularizedTest)
		{
			auto average_accuracy = 0.0;
			const bool long_test = false;
			const auto trials_count = 1;
			for (int i = 0; i < trials_count; i++)
				average_accuracy += RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, Real(0.2), long_test ? Real(0.97) : Real(0.95), long_test, Real(6.0));

			Logger::WriteMessage((std::string("Average accuracy = ") +
				std::to_string(average_accuracy/trials_count)).c_str());
		}

		TEST_METHOD(TrainingWithCrossEntropyCostAndTanhActivationTest)
		{
			const bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, Real(0.1), long_test ? Real(0.974) : Real(0.946), long_test, Real(0),
				{ActivationFunctionId::TANH, ActivationFunctionId::SIGMOID});
		}

		TEST_METHOD(TrainingWithCrossEntropyCostAndSoftMaxActivationTest)
		{
			const bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, Real(0.1), long_test ? Real(0.978) : Real(0.95), long_test, Real(0),
				{ ActivationFunctionId::SIGMOID, ActivationFunctionId::SOFTMAX });
		}

		TEST_METHOD(TrainingWithCrossEntropyCostAndReluActivationTest)
		{
			const bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, Real(0.075), long_test ? Real(0.977) : Real(0.95), long_test, Real(0),
				{ ActivationFunctionId::RELU, ActivationFunctionId::SIGMOID });
		}

		TEST_METHOD(NetSerializationTest)
		{
			//Arrange
			const auto in_dimension = 784;
			const auto out_dimension = 10;
			const auto net = Net(std::vector<std::size_t>{ in_dimension, 100, out_dimension });

			//Act
			const auto msg = MsgPack::pack(net);
			const auto net_unpacked = MsgPack::unpack<Net>(msg);

			//Assert
			const auto tests_samples_count = 100;
			for (int test_sample_id = 0; test_sample_id < tests_samples_count; test_sample_id++)
			{
				//take a random input sample
				const auto input_sample = Tensor(1, 1, in_dimension, -1, 1);
				const auto ref_output = net.act(input_sample);
				//Sanity check
				Assert::IsTrue(ref_output.max_abs() > 0 && input_sample.max_abs() > 0, L"Both input sample and reference output are expected to be non-zero.");

				const auto trial_output = net_unpacked.act(input_sample);

				Assert::IsTrue(ref_output == trial_output, L"Nets are not the same.");
			}
		}

		TEST_METHOD(NetScriptInstantiationTest)
		{
			//Arrange
			Net net;
			auto size_in_next = Index3d{1, 222, 333};
			const auto out_size = 10;
			size_in_next = net.append_layer<CLayer>(size_in_next, Index2d{ 5 }, 20, ActivationFunctionId::RELU);
			size_in_next = net.append_layer<PLayer>(size_in_next, Index2d{ 2 }, PoolTypeId::MAX);
			size_in_next = net.append_layer<CLayer>(size_in_next, Index2d{ 5 }, 10, ActivationFunctionId::RELU);
			size_in_next = net.append_layer<PLayer>(size_in_next, Index2d{ 2 }, PoolTypeId::MAX);
			size_in_next = net.append_layer<NLayer>(size_in_next.coord_prod(), 100, ActivationFunctionId::RELU, Real(-1), Real(1), true);
			size_in_next = net.append_layer<NLayer>(size_in_next.coord_prod(), out_size, ActivationFunctionId::SOFTMAX, Real(-1), Real(1), true);

			//Act
			const auto script_str = net.to_script();
			const auto net_restored = Net(script_str);

			//Assert
			Assert::IsTrue(net.equal_hyperparams(net_restored), L"Original and restored nets are different");
		}
	};
}
