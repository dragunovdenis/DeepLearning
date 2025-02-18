//Copyright (c) 2024 Denys Dragunov, dragunovdenis@gmail.com
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

#pragma once
#include "MNet.h"
#include "../MsgPackUtils.h"

namespace DeepLearning
{
	template <class D>
	void MNet<D>::evaluate_cost_gradient_in_place(IMLayerExchangeData<typename D::tensor_t>& in_out,
		const IMLayerExchangeData<typename D::tensor_t>& reference, const CostFunction<typename D::tensor_t>& cost_func)
	{
		for (auto item_id = 0ull; item_id < in_out.size(); ++item_id)
			cost_func.deriv_in_place(in_out[item_id], reference[item_id]);
	}

	template <class D>
	void MNet<D>::reset_gradients(std::vector<MLayerGradient<D>>& gradients)
	{
		for (auto& grad : gradients)
			grad.fill_zero();
	}

	template <class D>
	const LazyVector<typename D::tensor_t>& MNet<D>::Context::get_out() const
	{
		return value_cache.out();
	}

	template <class D>
	typename MNet<D>::Context MNet<D>::allocate_context() const
	{
		 Context result(_layers.size());
		 allocate_gradients(result.gradients, false /*fill zero*/);

		 result.layer_data_cache.shrink_to_fit();
		 for (auto layer_id = 0ull; layer_id < _layers.size(); ++layer_id)
			 result.layer_data_cache[layer_id] = _layers[layer_id].layer().allocate_trace_data();

		 return result;
	}

	template <class D>
	void MNet<D>::allocate_gradients(std::vector<MLayerGradient<D>>& gradients, const bool fill_zero) const
	{
		gradients.resize(_layers.size());
		gradients.shrink_to_fit();

		for (auto layer_id = 0ull; layer_id < _layers.size(); ++layer_id)
			gradients[layer_id] = _layers[layer_id].layer().allocate_gradient_container(fill_zero);
	}

	template <class D>
	std::vector<MLayerGradient<D>> MNet<D>::allocate_gradients(const bool fill_zero) const
	{
		std::vector<MLayerGradient<D>> result{};
		allocate_gradients(result, fill_zero);
		return result;
	}

	template <class D>
	void MNet<D>::act(const IMLayerExchangeData<typename D::tensor_t>& input, InOutMData<D>& cache) const
	{
		if (_layers.size() == 0)
			throw std::exception("The result is undefined.");

		_layers[0].layer().act(input, cache.out(), nullptr);

		for (auto layer_id = 1ull; layer_id < _layers.size(); ++layer_id)
		{
			cache.swap();
			_layers[layer_id].layer().act(cache.in(), cache.out(), nullptr);
		}
	}

	template <class D>
	LazyVector<typename D::tensor_t> MNet<D>::act(const IMLayerExchangeData<typename D::tensor_t>& input) const
	{
		thread_local InOutMData<D> cache{};
		act(input, cache);
		return cache.out();
	}

	template <class D>
	void MNet<D>::calc_gradient_sum(const LazyVector<LazyVector<typename D::tensor_t>>& input,
		const LazyVector<LazyVector<typename D::tensor_t>>& reference,
		const CostFunction<typename D::tensor_t>& cost_func, Context& context) const
	{
		if (_layers.size() == 0)
			throw std::exception("The result is undefined.");

		if (input.size() != reference.size())
			throw std::exception("Inconsistent input data.");

		reset_gradients(context.gradients);

		for (auto item_id = 0ull; item_id < input.size(); ++item_id)
			add_gradient(input[item_id], reference[item_id], cost_func, context);
	}

	template <class D>
	std::vector<MLayerGradient<D>> MNet<D>::calc_gradient_sum(
		const LazyVector<LazyVector<typename D::tensor_t>>& input,
		const LazyVector<LazyVector<typename D::tensor_t>>& reference,
		const CostFunction<typename D::tensor_t>& cost_func) const
	{
		auto context = allocate_context();
		calc_gradient_sum(input, reference, cost_func, context);
		return context.gradients;
	}

	template <class D>
	void MNet<D>::add_gradient(const LazyVector<typename D::tensor_t>& input, const LazyVector<typename D::tensor_t>& reference,
		const CostFunction<typename D::tensor_t>& cost_func, Context& context) const
	{
		auto& gradients = context.gradients;
		auto& aux = context.layer_data_cache;
		auto& deltas = context.value_cache;
		const auto layer_count = _layers.size();

		aux[0].assign_input(input);

		for (auto layer_id = 0ull; layer_id < layer_count; ++layer_id)
			_layers[layer_id].layer().act(aux[layer_id], aux[layer_id + 1], &aux[layer_id]);

		deltas.out().assign(aux[layer_count]);

		evaluate_cost_gradient_in_place(deltas.out(), reference, cost_func);

		for (long long layer_id = layer_count - 1; layer_id >= 0; --layer_id)
		{
			deltas.swap();
			_layers[layer_id].layer().backpropagate(deltas.in(), aux[layer_id + 1],
				aux[layer_id].Data, deltas.out(), gradients[layer_id], layer_id > 0);
		}
	}

	template <class D>
	std::mt19937& MNet<D>::ran_gen()
	{
		thread_local std::mt19937 ran_gen{ std::random_device{}() };
		return ran_gen;
	}

	template <class D>
	std::vector<MLayerGradient<D>> MNet<D>::calc_gradient(const LazyVector<typename D::tensor_t>& input,
		const LazyVector<typename D::tensor_t>& reference, const CostFunction<typename D::tensor_t>& cost_func) const
	{
		auto context = allocate_context();
		reset_gradients(context.gradients);
		add_gradient(input, reference, cost_func, context);
		return context.gradients;
	}

	template <class D>
	void MNet<D>::learn(const LazyVector<LazyVector<typename D::tensor_t>>& input,
		const LazyVector<LazyVector<typename D::tensor_t>>& reference,
		const CostFunction<typename D::tensor_t>& cost_func, const Real learning_rate, Context& context)
	{
		calc_gradient_sum(input, reference, cost_func, context);

		const auto scale = - learning_rate / input.size();
		update(context.gradients, scale);
	}

	template <class D>
	void MNet<D>::learn(const LazyVector<LazyVector<typename D::tensor_t>>& input,
		const LazyVector<LazyVector<typename D::tensor_t>>& reference,
		const CostFunction<typename D::tensor_t>& cost_func, const Real learning_rate)
	{
		auto context = allocate_context();
		learn(input, reference, cost_func, learning_rate, context);
	}

	template <class D>
	void MNet<D>::update(const std::vector<MLayerGradient<D>>& increments, const Real learning_rate)
	{
		if (increments.size() != _layers.size())
			throw std::exception("Invalid input.");

		for (auto layer_id = 0ull; layer_id < _layers.size(); ++layer_id)
			_layers[layer_id].layer().update(increments[layer_id], learning_rate);
	}

	template <class D>
	std::size_t MNet<D>::layer_count() const
	{
		return _layers.size();
	}

	template <class D>
	Index4d MNet<D>::in_size() const
	{
		if (_layers.empty())
			return { {0, 0, 0,}, 0 };

		return _layers.begin()->layer().in_size();
	}

	template <class D>
	Index4d MNet<D>::out_size() const
	{
		if (_layers.empty())
			return { {0, 0, 0,}, 0 };

		return _layers.rbegin()->layer().out_size();
	}

	template <class D>
	void MNet<D>::reset_random_generator(const unsigned seed)
	{
		ran_gen().seed(seed);
		AMLayer<D>::reset_random_generator(seed);
	}

	template <class D>
	void MNet<D>::reset_random_generator()
	{
		ran_gen().seed(std::random_device{}());
		AMLayer<D>::reset_random_generator();
	}

	template <class D>
	bool MNet<D>::operator==(const MNet& net) const
	{
		if (layer_count() != net.layer_count())
			return false;

		for (auto layer_id = 0ull; layer_id < layer_count(); ++layer_id)
		{
			if (!_layers[layer_id].layer().equal(net._layers[layer_id].layer()))
				return false;
		}

		return true;
	}

	template <class D>
	bool MNet<D>::operator!=(const MNet& net) const
	{
		return !(*this == net);
	}

	template <class D>
	void MNet<D>::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		auto msg_pack_version = 0;
		msgpack::type::make_define_array(msg_pack_version).msgpack_unpack(msgpack_o);

		if (msg_pack_version != MSG_PACK_VER)
			throw std::exception("Unexpected version of an object");

		msgpack::type::make_define_array(msg_pack_version, _layers).msgpack_unpack(msgpack_o);
	}

	template <class D>
	template <typename Packer>
	void MNet<D>::msgpack_pack(Packer& msgpack_pk) const
	{
		msgpack::type::make_define_array(MSG_PACK_VER, _layers).msgpack_pack(msgpack_pk);
	}

	template <class D>
	void MNet<D>::save_to_file(const std::filesystem::path& file_name) const
	{
		MsgPack::save_to_file(*this, file_name);
	}

	template <class D>
	MNet<D> MNet<D>::load_from_file(const std::filesystem::path& file_name)
	{
		return MsgPack::load_from_file<MNet>(file_name);
	}

	template <class D>
	template <template<class> class L, class... Types>
	Index4d MNet<D>::append_layer(Types&&... args)
	{
		auto new_layer = MLayerHandle<D>::template make<L>(std::forward<Types>(args)...);

		if (_layers.size() > 0 && _layers.rbegin()->layer().out_size() != new_layer.layer().in_size())
			throw std::exception("Incompatible input size of the new layer.");

		_layers.emplace_back(std::move(new_layer));
		return _layers.rbegin()->layer().out_size();
	}
}