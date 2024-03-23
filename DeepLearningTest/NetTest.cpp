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
#include <chrono>
#include "Utilities.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(NetTest)
	{
		/// <summary>
		/// A general method to run MNIST-based training and evaluation
		/// </summary>
		/// <param name="cost_func_id">Id of the cost function we want to use for training.</param>
		/// <param name="learning_rate">The learning rate we want to use.</param>
		/// <param name="expected_min_percentage_test_set">Expected minimal percentage of correct answers of the test data after the training.
		/// Can take values from (0, 1).</param>
		template <class D = CpuDC>
		static Real RunMnistBasedTrainingTest(const CostFunctionId cost_func_id, const Real& learning_rate,
			const Real& expected_min_percentage_test_set, const bool run_long_test, const Real& lambda = Real(0),
			const std::vector<ActivationFunctionId>& activ_func_ids = std::vector<ActivationFunctionId>())
		{
			//Arrange
			constexpr auto training_images_count = 60000;
			const auto [training_data, training_labels] = MnistDataUtils::load_labeled_data<D>(
				"TestData\\MNIST\\train-images.idx3-ubyte",
				"TestData\\MNIST\\train-labels.idx1-ubyte",
				training_images_count);

			constexpr auto test_images_count = 10000;
			const auto test_data_tuple = MnistDataUtils::load_labeled_data<D>(
				"TestData\\MNIST\\t10k-images.idx3-ubyte",
				"TestData\\MNIST\\t10k-labels.idx1-ubyte",
				test_images_count);

			const auto& test_data = std::get<0>(test_data_tuple);
			const auto& test_labels = std::get<1>(test_data_tuple);

			auto net = Net<D>({ 784, (run_long_test ? 100ull : 30ull), 10 }, activ_func_ids);
			constexpr auto batch_size = 10;
			const auto epochs_count = run_long_test ? 30 : 6;

			const auto evaluation_action = [&](const auto epoch_id, const auto scaled_l2_reg_factor)
			{
				const auto correct_answers = net.count_correct_answers(test_data, test_labels);
				Logger::WriteMessage((std::string("Epoch: ") + std::to_string(epoch_id) +  ". Correct answers : " + std::to_string(correct_answers) + "\n").c_str());
			};

			//Act
			const auto start = std::chrono::steady_clock::now();
			net.learn(training_data, training_labels, batch_size, epochs_count,
				learning_rate, cost_func_id, lambda, evaluation_action);
			const auto end = std::chrono::steady_clock::now();
			Logger::WriteMessage(("Learning time : " +
				std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) + " ms.").c_str());

			//Assert
			const auto correct_answers = net.count_correct_answers(test_data, test_labels);
			const auto validation_result = correct_answers * (Real(1)) / test_data.size();
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
		template <class D = CpuDC>
		static Real RunMnistBasedConvolutionNetTrainingTest(const CostFunctionId cost_func_id, const Real& learning_rate,
			const Real& expected_min_percentage_test_set, const bool run_long_test, const Real& lambda = Real(0))
		{
			//Arrange
			constexpr auto training_images_count = 60000;
			auto [training_data, training_labels] = MnistDataUtils::load_labeled_data<D>(
				"TestData\\MNIST\\train-images.idx3-ubyte",
				"TestData\\MNIST\\train-labels.idx1-ubyte",
				training_images_count);

			constexpr auto test_images_count = 10000;
			const auto test_data_tuple = MnistDataUtils::load_labeled_data<D>(
				"TestData\\MNIST\\t10k-images.idx3-ubyte",
				"TestData\\MNIST\\t10k-labels.idx1-ubyte",
				test_images_count);

			auto net = Net<D>();
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

			constexpr auto batch_size = 10;
			const auto epochs_count = run_long_test ? 30 : 5;

			std::chrono::steady_clock::time_point epoch_start;

			const auto& test_data = std::get<0>(test_data_tuple);
			const auto& test_labels = std::get<1>(test_data_tuple);

			const auto evaluation_action = [&](const auto epoch_id, const auto scaled_l2_reg_factor)
			{
				const auto correct_answers = net.count_correct_answers(test_data, test_labels);
				Logger::WriteMessage((std::string("Epoch: ") + std::to_string(epoch_id) + ". Correct answers : " + std::to_string(correct_answers)).c_str());
				auto epoch_end = std::chrono::steady_clock::now();
				Logger::WriteMessage((" Epoch time : " +
					std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start).count()) + " ms.\n").c_str());
				epoch_start = epoch_end;
			};

			//Act
			const auto start = std::chrono::steady_clock::now();
			epoch_start = start;
			net.learn(training_data, training_labels, batch_size, epochs_count,
				learning_rate, cost_func_id, lambda, evaluation_action);
			auto end = std::chrono::steady_clock::now();
			Logger::WriteMessage(("Learning time : " +
				std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) + " ms.").c_str());

			//Assert
			const auto correct_answers = net.count_correct_answers(test_data, test_labels);
			const auto validation_result = correct_answers * static_cast<Real>(1) / test_data.size();
			Assert::IsTrue(validation_result >= expected_min_percentage_test_set, L"Too low accuracy on the test set.");
			Logger::WriteMessage("\n");
			return validation_result;
		}


	public:

		TEST_METHOD(TrainingConvolutionNetWithCrossEntropyCostTest)
		{
			constexpr bool long_test = false;
			RunMnistBasedConvolutionNetTrainingTest(CostFunctionId::CROSS_ENTROPY, static_cast<Real>(0.025),
				long_test ? static_cast<Real>(0.991) : static_cast<Real>(0.98), long_test, static_cast<Real>(3));
		}

		//This test works but I am not satisfied with its performance (execution time), so it is out-commented for now
		//TEST_METHOD(CudaTrainingConvolutionNetWithCrossEntropyCostTest)
		//{
		//	constexpr bool long_test = false;
		//	RunMnistBasedConvolutionNetTrainingTest<GpuDC>(CostFunctionId::CROSS_ENTROPY, static_cast<Real>(0.03),
		//		long_test ? static_cast<Real>(0.991) : static_cast<Real>(0.98), long_test, static_cast<Real>(3));
		//}

		TEST_METHOD(TrainingWithQuadraticCostTest)
		{
			constexpr bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::SQUARED_ERROR, static_cast<Real>(1.0),
				long_test ? static_cast<Real>(0.976) : static_cast<Real>(0.95), long_test);
		}

		TEST_METHOD(TrainingWithQuadraticCostRegularizedTest)
		{
			constexpr bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::SQUARED_ERROR, static_cast<Real>(1.0),
				long_test ? static_cast<Real>(0.976) : static_cast<Real>(0.95), long_test, static_cast<Real>(1.0));
		}

		TEST_METHOD(TrainingWithCrossEntropyCostTest)
		{
			constexpr bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, static_cast<Real>(0.25),
				long_test ? static_cast<Real>(0.977) : static_cast<Real>(0.95), long_test);
		}

		TEST_METHOD(TrainingWithCrossEntropyCostRegularizedTest)
		{
			auto average_accuracy = 0.0;
			constexpr bool long_test = false;
			constexpr auto trials_count = 1;
			for (int i = 0; i < trials_count; i++)
				average_accuracy += RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, static_cast<Real>(0.2),
					long_test ? static_cast<Real>(0.97) : static_cast<Real>(0.95), long_test, static_cast<Real>(6.0));

			Logger::WriteMessage((std::string("Average accuracy = ") +
				std::to_string(average_accuracy/trials_count)).c_str());
		}

		TEST_METHOD(TrainingWithCrossEntropyCostAndTanhActivationTest)
		{
			constexpr bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, static_cast<Real>(0.1),
				long_test ? static_cast<Real>(0.974) : static_cast<Real>(0.945), long_test, static_cast<Real>(0),
				{ActivationFunctionId::TANH, ActivationFunctionId::SIGMOID});
		}

		TEST_METHOD(TrainingWithCrossEntropyCostAndSoftMaxActivationTest)
		{
			constexpr bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, static_cast<Real>(0.1),
				long_test ? static_cast<Real>(0.978) : static_cast<Real>(0.95), long_test, static_cast<Real>(0),
				{ ActivationFunctionId::SIGMOID, ActivationFunctionId::SOFTMAX });
		}

		//This test works but I am not satisfied with its performance (execution time), so it is out-commented for now
		//TEST_METHOD(CudaTrainingWithCrossEntropyCostAndSoftMaxActivationTest)
		//{
		//	const bool long_test = false;
		//	RunMnistBasedTrainingTest<GpuDC>(CostFunctionId::CROSS_ENTROPY, Real(0.1), long_test ? Real(0.978) : Real(0.95), long_test, Real(0),
		//		{ ActivationFunctionId::SIGMOID, ActivationFunctionId::SOFTMAX });
		//}

		TEST_METHOD(TrainingWithCrossEntropyCostAndReluActivationTest)
		{
			const bool long_test = false;
			RunMnistBasedTrainingTest(CostFunctionId::CROSS_ENTROPY, Real(0.075), long_test ? Real(0.977) : Real(0.95), long_test, Real(0),
				{ ActivationFunctionId::RELU, ActivationFunctionId::SIGMOID });
		}

		/// <summary>
		///	Generates and returns a "standard" net for testing
		/// </summary>
		static Net<CpuDC> GenerateStandardNet(const bool no_drop_out = false)
		{
			Net<CpuDC> net;
			constexpr auto layers_count = 7;
			std::mt19937 rg(0);
			const auto keep_rates = no_drop_out ? std::vector<Real>(layers_count, static_cast<Real>(1)) :
			Utils::get_random_std_vector(layers_count, 0, 1, rg);

			auto size_in_next = Index3d{ 1, 22, 33 };
			constexpr auto out_size = 10;
			size_in_next = net.append_layer<CLayer>(size_in_next, Index2d{ 5 }, 20, ActivationFunctionId::RELU,
				Index3d{ 0, 0, 0 }, Index3d{ 1, 1, 1 }, keep_rates[0]);
			size_in_next = net.append_layer<PLayer>(size_in_next, Index2d{ 2 }, PoolTypeId::MAX, keep_rates[1]);
			size_in_next = net.append_layer<CLayer>(size_in_next, Index2d{ 5 }, 10, ActivationFunctionId::TANH,
				Index3d{ 0, 0, 0 }, Index3d{ 1, 1, 1 }, keep_rates[2]);
			size_in_next = net.append_layer<PLayer>(size_in_next, Index2d{ 2 }, PoolTypeId::AVERAGE, keep_rates[3]);
			size_in_next = net.append_layer<NLayer>(size_in_next.coord_prod(), 10, ActivationFunctionId::LINEAR, Real(-1), Real(1), true, keep_rates[4]);
			size_in_next = net.append_layer<NLayer>(size_in_next.coord_prod(), 10, ActivationFunctionId::SIGMOID, Real(-1), Real(1), true, keep_rates[5]);
			size_in_next = net.append_layer<NLayer>(size_in_next.coord_prod(), out_size, ActivationFunctionId::SOFTMAX, Real(-1), Real(1), true, keep_rates[6]);

			for (auto layer_id = 0ull; layer_id < net.layers_count(); layer_id++)
				Assert::IsTrue(net[layer_id].get_keep_rate() == keep_rates[layer_id], L"Unexpected value of the `keep rate` parameter");

			return net;
		}

		TEST_METHOD(NetSerializationTest)
		{
			//Arrange
			const auto net = GenerateStandardNet();

			//Act
			const auto msg = MsgPack::pack(net);
			const auto net_unpacked = MsgPack::unpack<Net<CpuDC>>(msg);

			//Assert
			Assert::IsTrue(net.equal(net_unpacked), L"Original and restored nets are different");
		}

		TEST_METHOD(NetScriptInstantiationTest)
		{
			//Arrange
			const auto net = GenerateStandardNet();

			//Act
			const auto script_str = net.to_script();
			const auto net_restored = Net(script_str);

			//Assert
			Assert::IsTrue(net.equal_hyperparams(net_restored), L"Original and restored nets are different");
		}

		/// <summary>
		/// Generates artificial data set for training.
		/// </summary>
		template <class D>
		std::tuple<std::vector<typename D::tensor_t>, std::vector<typename D::tensor_t>> generate_artificial_training_data(
			const int items_count, const Index3d& in_size, const Index3d& out_size)
		{
			std::mt19937 rg{ 0 };
			std::vector<typename D::tensor_t> input;
			std::vector<typename D::tensor_t> labels;

			for (auto item_id = 0; item_id < items_count; ++item_id)
			{
				input.emplace_back(in_size, static_cast<Real>(-1), static_cast<Real>(1), &rg);
				labels.emplace_back(out_size, static_cast<Real>(-1), static_cast<Real>(1), &rg);
			}

			return std::make_tuple(input, labels);
		}

		TEST_METHOD(SingleItemLearnTest)
		{
			//Arrange
			auto net = GenerateStandardNet(/*no_drop_out*/ true);
			auto net_clone = MsgPack::unpack<Net<CpuDC>>(MsgPack::pack(net));
			constexpr auto learning_rate = static_cast<Real>(0.1);
			constexpr auto reg_factor = static_cast<Real>(1.5);
			constexpr auto cost_func_id = CostFunctionId::SQUARED_ERROR;

			Assert::IsTrue(net.equal(net_clone), L"Nets are supposed to be identical");

			const auto [input, labels] = generate_artificial_training_data<CpuDC>(1, net.in_size(), net.out_size());

			Assert::IsTrue(input[0].max_abs() > 0 && labels[0].max_abs() > 0,
				L"Neither input nor label items are supposed to be trivial");

			//Act
			net.learn(input[0], labels[0], learning_rate, cost_func_id, reg_factor);

			//Assert
			Assert::IsFalse(net.equal(net_clone), L"Nets are supposed to be different at this point");
			//now we run learning of the clone network through the "general" method and compare the result with the "original" net
			net_clone.learn(input, labels, 1, 1, learning_rate, cost_func_id, reg_factor);

			Assert::IsTrue(net.equal(net_clone), L"Nets after learning are supposed to be equal");
		}

		TEST_METHOD(NetLearnRegressionTest)
		{
			//Arrange
			Net<CpuDC>::reset_random_generator(0);
			auto net = GenerateStandardNet(/*no_drop_out*/ false);

			const auto [input, labels] =
				generate_artificial_training_data<CpuDC>(40, net.in_size(), net.out_size());

			//Act
			net.learn(input, labels, /*batch size*/10, /*epochs count*/ 3,
				/*learning rate*/ static_cast<Real>(0.1), /*const func*/CostFunctionId::SQUARED_ERROR,
				/*regularization*/static_cast<Real>(1.5), [](const auto a, const auto b) {}, /*single threaded*/ true);

			Net<CpuDC>::reset_random_generator();

			//Assert
			if (std::is_same_v<Real, double>)
			{
				//net.save_to_file("../../DeepLearningTest/TestData/Regression//reference_net_single.dat");
				//return;
				const auto reference_net = Net<CpuDC>::load_from_file("TestData\\Regression\\reference_net.dat");
				Assert::IsTrue(net.equal(reference_net), L"Nets are supposed to be equal.");
			} else
			{
				//net.save_to_file("../../DeepLearningTest/TestData/Regression//reference_net_single.dat");
				//return;
				const auto reference_net = Net<CpuDC>::load_from_file("TestData\\Regression\\reference_net_single.dat");
				Assert::IsTrue(net.equal(reference_net), L"Nets are supposed to be equal.");
			}
		}

		TEST_METHOD(LinearCostTest)
		{
			//Arrange
			Net<CpuDC> net;
			constexpr auto out_dim = 100;
			constexpr auto in_dim = 10;
			net.append_layer<NLayer>(in_dim, out_dim, ActivationFunctionId::LINEAR);

			const CpuDC::tensor_t input(1, 1, in_dim, -1, 1);
			const CpuDC::tensor_t target_value(1, 1, out_dim, -1, 1);
			Assert::IsTrue(input.max_abs() > 0 && target_value.max_abs() > 0, L"Vectors are supposed to be nonzero");

			//Make up the "biases derivative"
			CpuDC::tensor_t expected_bias_derivative(1, 1, out_dim);
			expected_bias_derivative.fill(1);

			//Make up the "weights derivative"
			const CpuDC::tensor_t expected_weight_derivative = vector_col_times_vector_row(expected_bias_derivative, input);

			//Act
			const auto gradient_and_value = net.calc_gradient_and_value(input, target_value, CostFunctionId::LINEAR);

			//Assert
			const auto gradient = std::get<0>(gradient_and_value);
			const auto weight_derivative = gradient[0].Weights_grad[0];
			const auto bias_derivative = gradient[0].Biases_grad;

			Assert::IsTrue((weight_derivative - expected_weight_derivative).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Too high deviation from the expected derivative with respect to the weight");

			Assert::IsTrue((bias_derivative - expected_bias_derivative).max_abs() < 10 * std::numeric_limits<Real>::epsilon(),
				L"Too high deviation from the expected derivative with respect to the bias");

			const auto value = std::get<1>(gradient_and_value);
			const auto value_ref = net.act(input);
			Assert::IsTrue(value == value_ref, L"Unexpected difference in the values of the net calculated via different interfaces");
		}

		TEST_METHOD(NetCopyTest)
		{
			//Arrange
			const auto net = GenerateStandardNet(/*no_drop_out*/ false);

			//Act
			const auto net_copy = net;

			//Assert
			Assert::IsTrue(net.equal(net_copy), L"Copying failed");
		}

		TEST_METHOD(NetResetTest)
		{
			//Arrange
			auto net = GenerateStandardNet();
			//Sanity check
			for (auto layer_id = 0ull; layer_id < net.layers_count(); ++layer_id)
			{
				if (net[layer_id].get_type_id() == LayerTypeId::PULL)
					continue;

				Assert::IsTrue(net[layer_id].squared_weights_sum() > 0, L"Weights are already zero");
			}

			//Act
			net.reset();

			//Assert
			//Simplified check (without biases), it is because we rely on the fact that net calls "reset" of all the layers.
			//The fact that weights are zero means that "reset" of the corresponding layer was actually called which
			//guarantees that both weights and biases are set to zero (see the corresponding tests for the "reset" functionality of layers)
			for (auto layer_id = 0ull; layer_id < net.layers_count(); ++layer_id)
				Assert::IsTrue(net[layer_id].squared_weights_sum() <= 0, L"Weights are non-zero");
		}

		/// <summary>
		/// Fills given gradient container with random values.
		/// </summary>
		static void randomize_gradient(std::vector<LayerGradient<CpuDC>>& gradient)
		{
			auto check = 0.0;
			for (auto& layer_grad : gradient)
			{
				layer_grad.Biases_grad.uniform_random_fill(-1, 1);

				for (auto& filter_grad : layer_grad.Weights_grad)
					filter_grad.uniform_random_fill(-1, 1);

				check = layer_grad.max_abs();


			}

			Assert::IsTrue(std::ranges::all_of(gradient,
				[](const auto& g) { return g.max_abs() > 0 ||
				g.Weights_grad.empty() && g.Biases_grad.empty(); }), L"There should not be zero gradients");
		}

		TEST_METHOD(NetGradientWithScalingTest)
		{
			////Arrange
			const auto net = GenerateStandardNet(/*no_drop_out*/ false);
			std::vector<LayerGradient<CpuDC>> out_gradient;
			net.allocate(out_gradient, /*fill zero*/ false);
			randomize_gradient(out_gradient);
			const auto in_gradient = out_gradient;

			Tensor out_value;
			Net<CpuDC>::Context context;

			constexpr auto cost_func_id = CostFunctionId::SQUARED_ERROR;
			const auto scale_factor = Utils::get_random(-1, 1);
			const auto [input, labels] =
				generate_artificial_training_data<CpuDC>(1, net.in_size(), net.out_size());

			Assert::IsTrue(input[0].max_abs() > 0 && labels[0].max_abs() > 0,
				L"Neither input nor label items are supposed to be trivial");

			//Act
			net.reset_random_generator(0); // for reproducible dropout
			net.calc_gradient_and_value(input[0], labels[0], cost_func_id, out_gradient, out_value, scale_factor, context);

			// Assert
			net.reset_random_generator(0); // for reproducible dropout
			const auto [pure_gradient_ref, value_ref] =
				net.calc_gradient_and_value(input[0], labels[0], cost_func_id);

			net.reset_random_generator();

			Assert::IsTrue(pure_gradient_ref.size() == net.layers_count(), L"Reference gradient has unexpected size.");
			Assert::IsTrue(out_gradient.size() == net.layers_count(), L"Trial gradient has unexpected size.");

			for (auto layer_id = 0ull; layer_id < net.layers_count(); ++layer_id)
			{
				const auto layer_gradient_diff = (out_gradient[layer_id] - (in_gradient[layer_id] * scale_factor + pure_gradient_ref[layer_id])).max_abs();
				Assert::IsTrue(layer_gradient_diff < 10 * std::numeric_limits<Real>::epsilon(), L"Too high deviation from the reference gradient");
			}

			Assert::IsTrue(value_ref == out_value, L"Net value should not be affected by scaling");
		}

		TEST_METHOD_CLEANUP(CleanupCheck)
		{
			const auto alive_instances = BasicCollection::get_total_instances_count();
			const auto occupied_memory = BasicCollection::get_total_allocated_memory();

			Logger::WriteMessage((std::to_string(alive_instances) + " alive instance(-s) of `BasicCollection` occupying "
				+ std::to_string(occupied_memory) + " byte(-s) of memory.\n").c_str());
		}
	};
}
