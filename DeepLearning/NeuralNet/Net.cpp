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

#include "Net.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <exception>
#include "CummulativeGradient.h"
#include <thread>
#include <ppl.h>
#include "../IndexIterator.h"

namespace DeepLearning
{
	Net::Net(const std::vector<std::size_t>& layer_dimensions, const ActivationFunctionId& activ_func_id)
	{
		if (layer_dimensions.size() <= 1)
			throw std::exception("Invalid collection of layer dimensions.");

		static std::random_device rd;
		static std::mt19937 gen(rd());

		for (std::size_t id = 1; id < layer_dimensions.size(); id++)
		{
			const auto in_dim = layer_dimensions[id - 1];
			const auto out_dim = layer_dimensions[id];

			std::normal_distribution<Real> dist{ 0, Real(1) / Real(std::sqrt(in_dim))};

			_layers.emplace_back(DenseMatrix(out_dim, in_dim,  [&]() {return dist(gen); }),
				DenseVector(out_dim, Real(-1), Real(1)), activ_func_id);
		}
	}

	DenseVector Net::act(const DenseVector& input, std::vector<NeuralLayer::AuxLearningData>* const aux_data_ptr) const
	{
		if (aux_data_ptr != nullptr && aux_data_ptr->size() != _layers.size())
			throw std::exception("Invalid auxiliary data.");

		auto result = input;
		for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
			result = _layers[layer_id].act(result, aux_data_ptr != nullptr ? &(*aux_data_ptr)[layer_id] : nullptr);

		return result;
	}

	/// <summary>
	/// Applies random permutation to the given collection of indices
	/// </summary>
	void apply_random_permutation(std::vector<std::size_t>& indices)
	{
		std::random_device rd;
		std::mt19937 g(rd());

		std::shuffle(indices.begin(), indices.end(), g);
	}

	/// <summary>
	/// Returns collection containing "elements_count" integer
	/// indices starting from "0" all the way to "elements_count - 1"
	/// </summary>
	std::vector<std::size_t> get_indices(const std::size_t elements_count)
	{
		std::vector<std::size_t> result(elements_count);
		std::iota(result.begin(), result.end(), 0);

		return result;
	}

	/// <summary>
	/// Returns collection of the gradient collectors that is "compatible" with the given collection of neural layers
	/// </summary>
	std::vector<CummulativeGradient> init_gradient_collectors(const std::vector<NeuralLayer>& layers)
	{
		std::vector<CummulativeGradient> result;

		for (std::size_t layer_id = 0; layer_id < layers.size(); layer_id++)
		{
			const auto& layer = layers[layer_id];
			result.emplace_back(layer.in_dim(), layer.out_dim());
		}

		return result;
	}

	/// <summary>
	/// Resets all the collectors in the given collection
	/// </summary>
	void reset_gradient_collectors(std::vector<CummulativeGradient>& collectors)
	{
		for (std::size_t collector_id = 0; collector_id < collectors.size(); collector_id++)
			collectors[collector_id].reset();
	}

	void Net::learn(const std::vector<DenseVector>& training_items, const std::vector<DenseVector>& reference_items,
		const std::size_t batch_size, const std::size_t epochs_count, const Real learning_rate, const CostFunctionId& cost_func_id,
		const Real& lambda, const std::function<void(std::size_t)>& epoch_callback)
	{
		if (training_items.size() != reference_items.size())
			throw std::exception("Incompatible collection of training and reference items.");

		if (training_items.size() == 0)
			return;

		const auto reg_factor = -learning_rate * lambda / training_items.size();

		const auto cost_function = CostFunction(cost_func_id);

		auto gradient_collectors = init_gradient_collectors(_layers);

		const auto physical_cores_count = std::thread::hardware_concurrency() / 2;
		//For some reason, this exact number of threads (when used in the parallel "for" loop below)
		//gives the best performance on a PC with i7-10750H (the only PC where this code was
		//tested so far). Whoever is reading this comment, feel free to try other numbers of the threads.
		const auto threads_to_use = std::max<int>(1, physical_cores_count - 1);

		std::mutex mutex;

		auto  data_index_mapping = get_indices(training_items.size());

		for (std::size_t epoch_id = 0; epoch_id < epochs_count; epoch_id++)
		{
			apply_random_permutation(data_index_mapping);

			std::size_t batch_start_elem_id = 0;
			while (batch_start_elem_id < training_items.size())
			{
				//Reset gradient collectors before each batch
				reset_gradient_collectors(gradient_collectors);

				//If there remains less than 1.5 * batch_size elements in the collection we take all of them as a single batch
				//This is aimed to ensure that an actual batch will always contain not less than half of the batch size elements
				//(provided, of course, that the training collection itself contains not less than half of the batch size elements)
				const long long batch_end_elem_id = (training_items.size() - batch_start_elem_id) < 1.5 * batch_size ? 
					training_items.size() : batch_start_elem_id + batch_size;

				Concurrency::simple_partitioner splitter((batch_end_elem_id - batch_start_elem_id) / threads_to_use);
				concurrency::parallel_for<std::size_t>(batch_start_elem_id, batch_end_elem_id,
					[&](const std::size_t elem_id)
					{
						const auto input_item_id = data_index_mapping[elem_id];
						const auto& input = training_items[input_item_id];
						const auto& reference = reference_items[input_item_id];
						auto aux_data_ptr = std::vector<NeuralLayer::AuxLearningData>(_layers.size());
						const auto output = act(input, &aux_data_ptr);
						auto [cost, gradient] = cost_function.func_and_deriv(output, reference);

						auto back_prop_out = std::vector<NeuralLayer::LayerGradient>(_layers.size());
						//Back-propagate through all the layers
						for (long long layer_id = _layers.size() - 1; layer_id >= 0; layer_id--)
							std::tie(gradient, back_prop_out[layer_id]) = _layers[layer_id].backpropagate(
								gradient, aux_data_ptr[layer_id], layer_id != 0);

						std::lock_guard guard(mutex);
						for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
							gradient_collectors[layer_id].Add(back_prop_out[layer_id].Weights_grad, back_prop_out[layer_id].Biases_grad);
					}, splitter);

				batch_start_elem_id = batch_end_elem_id;

				for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
					_layers[layer_id].update(gradient_collectors[layer_id].calc_average_grarient(-learning_rate), reg_factor);
			}

			epoch_callback(epoch_id);
		}
	}

	Real Net::evaluate_cost_function(const std::vector<DenseVector>& test_input,
		const std::vector<DenseVector>& reference_output, const CostFunctionId& cost_func_id) const
	{
		if (test_input.size() != reference_output.size())
			throw std::exception("Invalid input.");

		const auto cost_function = CostFunction(cost_func_id);

		const auto cost_sum = concurrency::parallel_reduce(IndexIterator<std::size_t>(0), IndexIterator<std::size_t>(test_input.size()), Real(0),
			[&](const auto& start_iter, const auto& end_iter, const auto& init_val)
			{
				auto result = init_val;
				const auto start_id = *start_iter;
				const auto end_id = *end_iter;
				for (auto i = start_id; i < end_id; i++)
					result += cost_function(act(test_input[i]), reference_output[i]);

				return result;
			}, std::plus<Real>());

		return cost_sum / test_input.size();
	}

	std::size_t Net::count_correct_answers(const std::vector<DenseVector>& test_input,
		const std::vector<DenseVector>& labels, const Real& min_answer_probability) const
	{
		if (test_input.size() != labels.size())
			throw std::exception("Invalid input.");

		const auto correct_answers = concurrency::parallel_reduce(IndexIterator<std::size_t>(0), IndexIterator<std::size_t>(test_input.size()), 0,
			[&](const auto& start_iter, const auto& end_iter, const auto& init_val)
			{
				auto result = init_val;
				const auto start_id = *start_iter;
				const auto end_id = *end_iter;
				for (auto test_item_id = start_id; test_item_id < end_id; test_item_id++)
				{
					const auto& test_item = test_input[test_item_id];
					const auto ref_answer = labels[test_item_id].max_element_id();
					const auto trial_label = act(test_item);
					//after normalization each element of the trial answer can be treated
					//as a probability of the corresponding class
					const auto trial_answer_normalized = trial_label * (Real(1) / trial_label.sum());
					const auto trial_answer = trial_answer_normalized.max_element_id();
					const auto trial_anwer_probability = trial_answer_normalized(trial_answer);

					if (trial_answer == ref_answer && trial_anwer_probability >= min_answer_probability)
						result++;
				}
				return result;
			}, std::plus<int>());

		return  correct_answers;
	}
}