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
#include <algorithm>
#include <exception>
#include "CummulativeGradient.h"
#include <thread>
#include <ppl.h>
#include "../IndexIterator.h"
#include "../Diagnostics/Logging.h"
#include <fstream>
#include "../ThreadPool.h"

namespace DeepLearning
{
	template <class D>
	Net<D>::Net(const std::vector<std::size_t>& layer_dimensions, const std::vector<ActivationFunctionId>& activ_func_ids)
	{
		const auto layers_count = layer_dimensions.size() - 1;
		if (layers_count <= 0)
			throw std::exception("Invalid collection of layer dimensions.");

		//The situation when collection "activ_func_ids" is empty should be interpreted as each layer having SIGMOID as its activation function
		const auto af_ids_local = activ_func_ids.empty() ? std::vector<ActivationFunctionId>(layers_count, ActivationFunctionId::SIGMOID) : activ_func_ids;

		if (layers_count != af_ids_local.size())
			throw std::exception("Invalid collection of activation function identifiers.");

		for (std::size_t id = 1; id < layer_dimensions.size(); id++)
		{
			const auto in_dim = layer_dimensions[id - 1];
			const auto out_dim = layer_dimensions[id];
			_layers.push_back(LayerHandle<D>::template make<NLayer<D>>(in_dim, out_dim, af_ids_local[id - 1], Real(-1), Real(1), true));
		}
	}

	template <class D>
	void Net<D>::try_load_from_script(const std::string& script)
	{
		_layers.clear();

		const auto layer_scripts = Utils::split_by_char(script, '\n');

		Index3d prev_layer_out_size = { -1, -1, -1 };
		for (const auto& script : layer_scripts)
		{
			auto script_normalized = Utils::normalize_string(script);
			const auto layer_type_id = parse_layer_type(Utils::extract_word(script_normalized));

			if (layer_type_id == LayerTypeId::UNKNOWN)
				throw std::exception("Unknown layer type");

			if (prev_layer_out_size.x >= 0)
			{
				Index3d in_size;
				if (Utils::try_extract_vector(script_normalized, in_size))
				{
					if ((layer_type_id != LayerTypeId::FULL && prev_layer_out_size != in_size) ||
						(layer_type_id == LayerTypeId::FULL && prev_layer_out_size.coord_prod() != in_size.coord_prod()))
						throw std::exception((std::string("Inconsistent input/output dimensions : ") + script_normalized).c_str());
				}

				//This covers also a situation when script omits input size of the layer so that
				//it is deduced from the output size of the previous layer
				script_normalized = prev_layer_out_size.to_string() + script_normalized;
			}

			switch (layer_type_id)
			{
			case LayerTypeId::CONVOLUTION: append_layer<CLayer>(script_normalized); break;
			case LayerTypeId::FULL: append_layer<NLayer>(script_normalized); break;
			case LayerTypeId::PULL: append_layer<PLayer>(script_normalized); break;
			default:
				throw std::exception("Unexpected identifier of the layer type");
			}

			prev_layer_out_size = _layers.rbegin()->layer().out_size();
		}
	}

	template <class D>
	void Net<D>::try_load_from_script_file(const std::filesystem::path& file_path)
	{
		try_load_from_script(Utils::read_all_text(file_path));
	}

	template <class D>
	Net<D>::Net(const std::string& script_str)
	{
		try_load_from_script(script_str);
	}

	template <class D>
	void Net<D>::act(const typename D::tensor_t& input, InOutData& eval_data, std::vector<typename ALayer<D>::AuxLearningData>* const aux_data_ptr) const
	{
		if (aux_data_ptr != nullptr && aux_data_ptr->size() != _layers.size())
			throw std::exception("Invalid auxiliary data.");

		const auto training = aux_data_ptr != nullptr;
		eval_data.Out = input;
		for (auto layer_id = 0ull; layer_id < _layers.size(); layer_id++)
		{
			eval_data.swap();
			_layers[layer_id].layer().ApplyDropout(eval_data.In, training);
			_layers[layer_id].layer().act(eval_data.In, eval_data.Out, training ? &(*aux_data_ptr)[layer_id] : nullptr);
		}
	}

	template <class D>
	typename D::tensor_t Net<D>::act(const typename D::tensor_t& input, std::vector<typename ALayer<D>::AuxLearningData>* const aux_data_ptr) const
	{
		InOutData evalData;
		act(input, evalData, aux_data_ptr);

		return evalData.Out;
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
	template <class D>
	std::vector<CummulativeGradient<D>> init_gradient_collectors(const std::vector<LayerHandle<D>>& layers)
	{
		std::vector<CummulativeGradient<D>> result;

		for (std::size_t layer_id = 0; layer_id < layers.size(); layer_id++)
		{
			const auto& layer = layers[layer_id].layer();
			result.emplace_back(layer.init_cumulative_gradient());
		}

		return result;
	}

	/// <summary>
	/// Resets all the collectors in the given collection
	/// </summary>
	template <class D>
	void reset_gradient_collectors(std::vector<CummulativeGradient<D>>& collectors)
	{
		for (std::size_t collector_id = 0; collector_id < collectors.size(); collector_id++)
			collectors[collector_id].reset();
	}

	template <class D>
	void Net<D>::learn(const std::vector<typename D::tensor_t>& training_items, const std::vector<typename D::tensor_t>& reference_items,
		const std::size_t batch_size, const std::size_t epochs_count, const Real learning_rate, const CostFunctionId& cost_func_id,
		const Real& lambda, const std::function<void(const std::size_t, const Real)>& epoch_callback)
	{
		if (training_items.size() != reference_items.size())
			throw std::exception("Incompatible collection of training and reference items.");

		if (training_items.size() == 0)
			return;

		const auto lambda_scaled = lambda / training_items.size();
		const auto reg_factor = -learning_rate * lambda_scaled;

		const auto cost_function = CostFunction<typename D::tensor_t>(cost_func_id);

		auto gradient_collectors = init_gradient_collectors(_layers);

		const auto physical_cores_count = std::thread::hardware_concurrency() / 2;
		//For some reason, this exact number of threads (when used in the parallel "for" loop below)
		//gives the best performance on a PC with i7-10750H (the only PC where this code was
		//tested so far). Whoever is reading this comment, feel free to try other numbers of the threads.
		const auto threads_to_use = std::max<int>(1, physical_cores_count);

		ThreadPool thread_pool(threads_to_use);

		auto aux_learning_data = std::vector<std::vector<typename ALayer<D>::AuxLearningData>>(threads_to_use, std::vector<typename ALayer<D>::AuxLearningData>(_layers.size()));
		auto layer_gradient_data = std::vector<std::vector<LayerGradient<D>>>(threads_to_use, std::vector<LayerGradient<D>>(_layers.size()));
		auto eval_data = std::vector<InOutData>(threads_to_use);
		std::mutex mutex;

		auto  data_index_mapping = get_indices(training_items.size());

		for (std::size_t epoch_id = 0; epoch_id < epochs_count; ++epoch_id)
		{
			apply_random_permutation(data_index_mapping);

			std::size_t batch_start_elem_id = 0;
			while (batch_start_elem_id < training_items.size())
			{
				//Reset gradient collectors before each batch
				reset_gradient_collectors(gradient_collectors);

				//Generate new drop-out masks
				for (auto layer_id = 0ull; layer_id < _layers.size(); ++layer_id)
					_layers[layer_id].layer().SetUpDropoutMask();

				//If there remains less than 1.5 * batch_size elements in the collection we take all of them as a single batch
				//This is aimed to ensure that an actual batch will always contain not less than half of the batch size elements
				//(provided, of course, that the training collection itself contains not less than half of the batch size elements)
				const auto batch_end_elem_id = (training_items.size() - batch_start_elem_id) < 1.5 * batch_size ? 
					training_items.size() : batch_start_elem_id + batch_size;

				const auto items_per_thread = (batch_end_elem_id - batch_start_elem_id + threads_to_use - 1) / threads_to_use;
				auto currnt_thread_start_id = batch_start_elem_id;
				auto actual_jobs_count = 0;

				while (currnt_thread_start_id < batch_end_elem_id)
				{
					const auto current_thread_end_id = std::min(currnt_thread_start_id + items_per_thread, batch_end_elem_id);

					thread_pool.queue_job([&, currnt_thread_start_id, current_thread_end_id](const std::size_t local_thread_id)
						{
							auto& aux_data = aux_learning_data[local_thread_id];
							auto& e_data = eval_data[local_thread_id];
							auto& back_prop_out = layer_gradient_data[local_thread_id];
							for (auto elem_id = currnt_thread_start_id; elem_id < current_thread_end_id; elem_id++)
							{
								const auto input_item_id = data_index_mapping[elem_id];
								const auto& input = training_items[input_item_id];
								const auto& reference = reference_items[input_item_id];
								act(input, e_data, &aux_data);
								cost_function.deriv_in_place(e_data.Out, reference);

								//Back-propagate through all the layers
								for (long long layer_id = _layers.size() - 1; layer_id >= 0; --layer_id)
								{
									e_data.swap();
									_layers[layer_id].layer().backpropagate(e_data.In, aux_data[layer_id],
										e_data.Out, back_prop_out[layer_id], layer_id != 0);

									if (layer_id != 0)
										_layers[layer_id].layer().ApplyDropout(e_data.Out, true);
								}

								std::lock_guard guard(mutex);
								for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
									gradient_collectors[layer_id].add(back_prop_out[layer_id]);
							}
						});

					currnt_thread_start_id = current_thread_end_id;
					++actual_jobs_count;
				}

				thread_pool.wait_until_jobs_done(actual_jobs_count);

				batch_start_elem_id = batch_end_elem_id;

				for (std::size_t layer_id = 0; layer_id < _layers.size(); layer_id++)
					_layers[layer_id].layer().update(gradient_collectors[layer_id].calc_average_gradient(-learning_rate), reg_factor);
			}

			epoch_callback(epoch_id, Real(0.5) * lambda_scaled);
		}

		for (auto layer_id = 0ull; layer_id < _layers.size(); ++layer_id)
			_layers[layer_id].layer().DisposeDropoutMask();
	}

	template <class D>
	std::vector<LayerGradient<D>> Net<D>::calc_gradient(
		const typename D::tensor_t& training_item, const typename D::tensor_t& target_value, const CostFunctionId& cost_func_id)
	{
		const auto cost_function = CostFunction<typename D::tensor_t>(cost_func_id);

		//Generate drop-out masks (if required by the settings of each particular layer)
		for (auto layer_id = 0ull; layer_id < _layers.size(); ++layer_id)
			_layers[layer_id].layer().SetUpDropoutMask();

		std::vector<typename ALayer<D>::AuxLearningData> aux_data(_layers.size());
		std::vector<LayerGradient<D>> result(_layers.size());
		InOutData e_data{};

		//Forward move
		act(training_item, e_data, &aux_data);
		cost_function.deriv_in_place(e_data.Out, target_value);

		//Back-propagate through all the layers
		for (long long layer_id = _layers.size() - 1; layer_id >= 0; --layer_id)
		{
			e_data.swap();
			_layers[layer_id].layer().backpropagate(e_data.In, aux_data[layer_id],
				e_data.Out, result[layer_id], layer_id != 0);

			if (layer_id != 0)
				_layers[layer_id].layer().ApplyDropout(e_data.Out, true);
		}

		//Dispose auxiliary data structures created to do the "drop-out" regularization
		for (auto layer_id = 0ull; layer_id < _layers.size(); ++layer_id)
			_layers[layer_id].layer().DisposeDropoutMask();

		return result;
	}

	template <class D>
	void Net<D>::update(const std::vector<LayerGradient<D>>& gradient, const Real learning_rate, const Real& lambda)
	{
		const auto reg_factor = -learning_rate * lambda;
		for (auto layer_id = 0ull; layer_id < _layers.size(); ++layer_id)
		{
			_layers[layer_id].layer().update(gradient[layer_id] * (-learning_rate), reg_factor);
		}
	}

	template <class D>
	void Net<D>::learn(const typename D::tensor_t& training_item, const typename D::tensor_t& target_value,
		const Real learning_rate, const CostFunctionId& cost_func_id, const Real& lambda)
	{
		const auto gradient = calc_gradient(training_item, target_value, cost_func_id);
		update(gradient, learning_rate, lambda);
	}

	template <class D>
	Real Net<D>::squared_weights_sum() const
	{
		return std::accumulate(_layers.begin(), _layers.end(), Real(0),
			[](const auto& sum, const auto& layer_handle) { return sum + layer_handle.layer().squared_weights_sum(); });
	}

	template <class D>
	CostAndCorrectAnswers Net<D>::evaluate_cost_function_and_correct_answers(const std::vector<typename D::tensor_t>& test_input,
		const std::vector<typename D::tensor_t>& reference_output, const CostFunctionId& cost_func_id, const Real l2_reg_factor) const
	{
		if (test_input.size() != reference_output.size())
			throw std::exception("Invalid input.");

		const auto cost_function = CostFunction<typename D::tensor_t>(cost_func_id);

		auto cost_and_answers_accumulated = concurrency::parallel_reduce(IndexIterator<std::size_t>(0), IndexIterator<std::size_t>(test_input.size()), CostAndCorrectAnswers{},
			[&](const auto& start_iter, const auto& end_iter, const auto& init_val)
			{
				auto result = init_val;
				const auto start_id = *start_iter;
				const auto end_id = *end_iter;
				for (auto i = start_id; i < end_id; i++)
				{
					const auto trial_label = act(test_input[i]);
					result.Cost += cost_function(trial_label, reference_output[i]);

					const auto trial_answer = trial_label.max_element_id();
					const auto ref_answer = reference_output[i].max_element_id();
					result.CorrectAnswers += trial_answer == ref_answer;
				}

				return result;
			}, std::plus<CostAndCorrectAnswers>());

		cost_and_answers_accumulated.Cost = cost_and_answers_accumulated.Cost / test_input.size() + l2_reg_factor * squared_weights_sum();
		cost_and_answers_accumulated.CorrectAnswers /= test_input.size();

		return cost_and_answers_accumulated;
	}

	template <class D>
	std::size_t Net<D>::count_correct_answers(const std::vector<typename D::tensor_t>& test_input,
		const std::vector<typename D::tensor_t>& labels) const
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
					const auto trial_answer = trial_label.max_element_id();

					if (trial_answer == ref_answer)
						result++;
				}
				return result;
			}, std::plus<int>());

		return  correct_answers;
	}

	template <class D>
	const ALayer<D>& Net<D>::operator [](const std::size_t& id) const
	{
		return _layers[id].layer();
	}

	template <class D>
	ALayer<D>& Net<D>::operator [](const std::size_t& id)
	{
		return _layers[id].layer();
	}

	template <class D>
	std::size_t Net<D>::layers_count() const
	{
		return _layers.size();
	}

	template <class D>
	void Net<D>::log(const std::filesystem::path& directory) const
	{
		for (auto layer_id = 0ull; layer_id < _layers.size(); layer_id++)
		{
			const auto layer_directory = directory / (std::string("layer_") + std::to_string(layer_id));
			Logging::make_path(layer_directory);

			_layers[layer_id].layer().log(layer_directory);
		}
	}

	template <class D>
	std::string Net<D>::to_script() const
	{
		std::string result;

		for (const auto& layer_handel : _layers)
			result +=  DeepLearning::to_string(layer_handel.layer().get_type_id()) + ";" + layer_handel.layer().to_script() + '\n';

		return result;
	}

	template <class D>
	void Net<D>::save_script(const std::filesystem::path& script_path) const
	{
		std::ofstream file(script_path);

		if (!file.is_open())
			throw std::exception("Can't create file");

		file << to_script();
	}

	template <class D>
	bool Net<D>::equal_hyperparams(const Net<D>& net) const
	{
		if (_layers.size() != net._layers.size())
			return false;

		for (auto layer_id = 0ull; layer_id < _layers.size(); layer_id++)
			if (!_layers[layer_id].layer().equal_hyperparams(net._layers[layer_id].layer()))
				return false;

		return true;
	}

	template <class D>
	bool Net<D>::equal(const Net& net) const
	{
		if (_layers.size() != net._layers.size())
			return false;

		for (auto layer_id = 0ull; layer_id < _layers.size(); layer_id++)
			if (!_layers[layer_id].layer().equal(net._layers[layer_id].layer()))
				return false;

		return true;
	}

	template <class D>
	std::string Net<D>::to_string() const
	{
		std::string result{};

		for (const auto& layer_handle : _layers)
			result += layer_handle.layer().to_string() + "\n";

		return result;
	}

	template <class D>
	Index3d Net<D>::in_size() const
	{
		if (_layers.empty())
			return { -1, -1, -1 };

		return _layers[0].layer().in_size();
	}

	template <class D>
	Index3d Net<D>::out_size() const
	{
		if (_layers.empty())
			return { -1, -1, -1 };

		return _layers.rbegin()[0].layer().out_size();
	}

	template class Net<CpuDC>;
	template class Net<GpuDC>;

	CostAndCorrectAnswers& CostAndCorrectAnswers::operator += (const CostAndCorrectAnswers& item)
	{
		Cost += item.Cost;
		CorrectAnswers += item.CorrectAnswers;

		return *this;
	}

	CostAndCorrectAnswers operator +(const CostAndCorrectAnswers& item1, const CostAndCorrectAnswers& item2)
	{
		auto result = item1;
		return result += item2;
	}
}