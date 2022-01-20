#include "Net.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <exception>

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

			_layers.emplace_back(in_dim, out_dim, activ_func_id, false /*enable learning*/);
		}
	}

	DenseVector Net::act(const DenseVector& input) const
	{
		auto result = input;
		for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
			result = _layers[layer_id].act(result);

		return input;
	}

	void Net::SetLearningMode(const bool do_learning)
	{
		for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
			_layers[layer_id].enable_learning_mode(do_learning);
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
	std::vector<CummulativeLayerGradient> init_gradient_collectors(const std::vector<NeuralLayer>& layers)
	{
		std::vector<CummulativeLayerGradient> result;

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
	void reset_gradient_collectors(std::vector<CummulativeLayerGradient>& collectors)
	{
		for (std::size_t collector_id = 0; collector_id < collectors.size(); collector_id++)
			collectors[collector_id].reset();
	}

	template <class T>
	std::vector<Real> Net::learn(const std::vector<T>& training_items, const std::vector<DenseVector>& reference_items,
		const std::size_t batch_size, const std::size_t epochs_count, const Real learning_rate, const CostFunctionId& cost_func_id)
	{
		if (training_items.size() != reference_items.size())
			throw std::exception("Incompatible collection of training and reference items.");

		//Activate learning mode of each layer
		SetLearningMode(true);

		const auto cost_function = CostFunction(cost_func_id);

		auto gradient_collectors = init_gradient_collectors(_layers);

		std::vector<Real> result(epochs_count);
		for (std::size_t epoch_id = 0; epoch_id < epochs_count; epoch_id++)
		{
			const auto id_permutation = get_index_permutation(training_items.size());

			std::size_t batch_start_elem_id = 0;
			while (batch_start_elem_id < training_items.size())
			{
				//Reset gradient collectors before each batch
				reset_gradient_collectors(gradient_collectors);

				for (std::size_t elem_id = batch_start_elem_id;
					elem_id < training_items.size() && ((elem_id - batch_start_elem_id) < batch_size);
					elem_id++)
				{
					const auto input_item_id = id_permutation[elem_id];
					const auto& input = training_items[input_item_id];
					const auto& reference = reference_items[input_item_id];
					const auto output = act(input);
					auto [cost, gradient] = cost_function.func_and_deriv(output, reference);

					//Back-propagate through all the layers
					for (long long layer_id = _layers.size() - 1; layer_id >= 0; layer_id--)
						gradient = _layers[layer_id].backpropagate(gradient, gradient_collectors[layer_id]);
				}

				for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
					_layers[layer_id].update(gradient_collectors[layer_id].calc_average_grarient(-learning_rate));
			}

			Real cost = Real(0);
			//Evaluate cost function on the training set
			for (std::size_t item_id = 0; item_id < training_items.size(); item_id++)
				cost += cost_function(act(training_items[item_id]), reference_items[item_id]);

			cost /= training_items.size();

			result[epoch_id] = cost;
		}

		SetLearningMode(false);
		return result;
	}


}