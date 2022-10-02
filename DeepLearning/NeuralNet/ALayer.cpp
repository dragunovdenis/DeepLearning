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

#include "ALayer.h"

namespace DeepLearning
{
	/// <summary>
	///	Tag used in the script representation of the layer to denote its "one minus drop-out rate" value
	/// </summary>
	const std::string KeepTag = "KEEP";

	template <class D>
	void ALayer<D>::SetUpDropoutMask()
	{
		if (_keep_rate < Real(1)) // do this only if we actually want to drop something out
		{
			const auto input_linear_size = in_size().coord_prod();
			const auto selected_cnt = static_cast<std::size_t>(input_linear_size * _keep_rate);
			_keep_mask.resize(input_linear_size);
			_keep_mask.fill_with_random_selection_map(selected_cnt, _keep_mask_aux_collection);
		}
	}

	template <class D>
	void ALayer<D>::DisposeDropoutMask()
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
	ALayer<D>::ALayer(const Real keep_rate) : _keep_rate(keep_rate)
	{}

	template <class D>
	Real ALayer<D>::get_keep_rate() const
	{
		return _keep_rate;
	}

	template <class D>
	ALayer<D>::ALayer(const std::string& script)
	{
		auto script_normalized = Utils::normalize_string(script);
		const auto dropout_block_start = script_normalized.find(KeepTag);

		if (dropout_block_start == std::string::npos)
			_keep_rate = DefaultKeepRate;
		else
		{
			_keep_rate = Utils::str_to_float<Real>(
				Utils::extract_word(script_normalized, dropout_block_start + KeepTag.size()));
		}
	}

	template <class D>
	bool ALayer<D>::equal_hyperparams(const ALayer<D>& layer) const
	{
		return _keep_rate == layer._keep_rate;
	}

	template <class D>
	std::string ALayer<D>::to_script() const
	{
		return KeepTag + " " + Utils::float_to_str_exact(_keep_rate) + ";";
	};

	template <class D>
	std::string ALayer<D>::to_string() const
	{
		return KeepTag + ": " + Utils::to_string(_keep_rate) + ";";
	}

	template class ALayer<CpuDC>;
	template class ALayer<GpuDC>;
}