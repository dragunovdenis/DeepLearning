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

namespace DeepLearning
{
	Net::Net(const std::vector<std::size_t>& layer_dimensions, const ActivationFunctionId& activ_func_id)
	{
		if (layer_dimensions.size() <= 1)
			throw std::exception("Invalid collection of layer dimensions.");

		for (std::size_t id = 1; id < layer_dimensions.size(); id++)
		{
			const auto in_dim = layer_dimensions[id - 1];
			const auto out_dim = layer_dimensions[id];

			_layers.emplace_back(in_dim, out_dim, activ_func_id, Real(-1), Real(1));
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
	/// Returns random permutation of indices from 0 to elements_count - 1
	/// </summary>
	std::vector<std::size_t> get_index_permutation(const std::size_t elements_count)
	{
		std::vector<std::size_t> result(elements_count);
		std::iota(result.begin(), result.end(), 0);

		std::random_device rd;
		std::mt19937 g(rd());

		std::shuffle(result.begin(), result.end(), g);

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

	std::vector<Real> Net::learn(const std::vector<DenseVector>& training_items, const std::vector<DenseVector>& reference_items,
		const std::size_t batch_size, const std::size_t epochs_count, const Real learning_rate, const CostFunctionId& cost_func_id,
		const std::size_t cost_evaluation_step)
	{
		if (training_items.size() != reference_items.size())
			throw std::exception("Incompatible collection of training and reference items.");

		if (training_items.size() == 0)
			return std::vector<Real>();

		const auto cost_function = CostFunction(cost_func_id);

		auto gradient_collectors = init_gradient_collectors(_layers);

		const auto physical_cores_count = std::thread::hardware_concurrency() / 2;
		//For some reason, this exact number of threads (when used in the parallel "for" loop below)
		//gives the best performance on a PC with i7-10750H (the only PC where this code was
		//tested so far). Whoever is reading this comment, feel free to try other numbers of the threads.
		const auto threads_to_use = std::max<int>(1, physical_cores_count - 1);

		std::vector<Real> result;
		for (std::size_t epoch_id = 0; epoch_id < epochs_count; epoch_id++)
		{
			const auto id_permutation = get_index_permutation(training_items.size());

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

				#pragma omp parallel for num_threads(threads_to_use)
				for (long long elem_id = batch_start_elem_id; elem_id < batch_end_elem_id; elem_id++)
				{
					const auto input_item_id = id_permutation[elem_id];
					const auto& input = training_items[input_item_id];
					const auto& reference = reference_items[input_item_id];
					auto aux_data_ptr = std::vector<NeuralLayer::AuxLearningData>(_layers.size());
					const auto output = act(input, &aux_data_ptr);
					auto [cost, gradient] = cost_function.func_and_deriv(output, reference);

					auto back_prop_out = std::vector<NeuralLayer::LayerGradient>(_layers.size());
					//Back-propagate through all the layers
					for (long long layer_id = _layers.size() - 1; layer_id >= 0; layer_id--)
						std::tie(gradient, back_prop_out[layer_id]) = _layers[layer_id].backpropagate(gradient, aux_data_ptr[layer_id]);

					#pragma omp critical
					for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
						gradient_collectors[layer_id].Add(back_prop_out[layer_id].Weights_grad, back_prop_out[layer_id].Biases_grad);
				}

				batch_start_elem_id = batch_end_elem_id;

				for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
					_layers[layer_id].update(gradient_collectors[layer_id].calc_average_grarient(-learning_rate));
			}

			if (epoch_id % cost_evaluation_step != 0)
				continue;

			Real cost = Real(0);
			//Evaluate cost function on the training set
			#pragma omp parallel for reduction(+:cost)
			for (auto item_id = 0; item_id < training_items.size(); item_id++)
				cost += cost_function(act(training_items[item_id]), reference_items[item_id]);

			cost /= training_items.size();

			result.push_back(cost);
		}

		return result;
	}
}