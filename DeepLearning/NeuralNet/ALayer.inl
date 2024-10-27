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

#pragma once

#include "../Utilities.h"
#include <nlohmann/json.hpp>

namespace DeepLearning
{
	template <class D>
	void ALayer<D>::SetUpDropoutMask() const
	{
		if (_keep_rate < Real(1)) // do this only if we actually want to drop something out
		{
			const auto input_linear_size = in_size().coord_prod();
			const auto selected_cnt = static_cast<std::size_t>(input_linear_size * _keep_rate);
			_keep_mask.resize(input_linear_size);
			_keep_mask.fill_with_random_selection_map(selected_cnt, _keep_mask_aux_collection, &ran_gen());
		}
	}

	template <class D>
	void ALayer<D>::DisposeDropoutMask() const
	{
		//relying on the move assignment operator
		_keep_mask = typename D::vector_t();
		_keep_mask_aux_collection = typename D::template index_array_t<int>();
	}

	template <class D>
	void ALayer<D>::ApplyDropout(typename D::tensor_t& input, const bool trainingMode) const
	{
		if (_keep_rate >= Real(1))
			return;

		if (trainingMode)
			input.hadamard_prod_in_place(_keep_mask);
		else
			input.mul(_keep_rate);
	}

	template <class D>
	ALayer<D>::ALayer(const Real keep_rate, const ActivationFunctionId func_id) : _keep_rate(keep_rate)
	{
		set_func_id(func_id);
	}

	template <class D>
	std::mt19937& ALayer<D>::ran_gen()
	{
		thread_local std::mt19937 ran_gen{ std::random_device{}() };
		return ran_gen;
	}

	template <class D>
	const AFunction<typename D::tensor_t>& ALayer<D>::get_func() const
	{
		return *_func;
	}

	template <class D>
	void ALayer<D>::set_func_id(const ActivationFunctionId func_id)
	{
		_func_id = func_id;
		if (_func_id != ActivationFunctionId::UNKNOWN)
			_func = ActivationWrapper<typename D::tensor_t >::construct(_func_id);
		else
			_func.reset();
	}

	template <class D>
	void ALayer<D>::set_keep_rate(const Real keep_rate)
	{
		_keep_rate = keep_rate;
	}

	template <class D>
	void ALayer<D>::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		ActivationFunctionId func_id;
		auto msg_pack_version = 0;
		msgpack::type::make_define_array(msg_pack_version, _keep_rate, func_id).msgpack_unpack(msgpack_o);
		set_func_id(func_id);
	}

	template <class D>
	Real ALayer<D>::get_keep_rate() const
	{
		return _keep_rate;
	}

	template <class D>
	ALayer<D>::ALayer(const std::string& script)
	{
		const auto json = nlohmann::json::parse(script);

		if (json.contains(json_keep_id()))
			_keep_rate = json[json_keep_id()].template get<Real>();
		else
			_keep_rate = DefaultKeepRate;

		if (json.contains(json_activation_id()))
		{
			const auto func_id = parse_activation_type(json[json_activation_id()].template get<std::string>());
			set_func_id(func_id);
		}
		else
			set_func_id(ActivationFunctionId::UNKNOWN);
	}

	template <class D>
	std::string ALayer<D>::to_script() const
	{
		nlohmann::json json;
		json[json_keep_id()] = _keep_rate;
		json[json_activation_id()] = DeepLearning::to_string(_func_id);

		return json.dump();
	};

	template <class D>
	bool ALayer<D>::equal_hyperparams(const ALayer<D>& layer) const
	{
		return _keep_rate == layer._keep_rate && _func_id == layer._func_id;
	}

	template <class D>
	void ALayer<D>::reset_random_generator(const unsigned seed)
	{
		ran_gen().seed(seed);
	}

	template <class D>
	void ALayer<D>::reset_random_generator()
	{
		ran_gen().seed(std::random_device{}());
	}

	template <class D>
	std::string ALayer<D>::to_string() const
	{
		return to_script();
	}
}