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
#include "DataContext.h"
#include "MLayerHandle.h"
#include "InOutMData.h"
#include "MLayerData.h"
#include "../Math/LinAlg4d.h"
#include "../Math/CostFunction.h"

namespace DeepLearning
{
	/// <summary>
	/// Implementation of a multi-net.
	/// </summary>
	template <class D = CpuDC>
	class MNet
	{
	public:

		/// <summary>
		/// Data structure that represents auxiliary resources needed to do inferring/training of the neural net.
		/// </summary>
		class Context
		{
			friend class MNet;//keep everything private and visible only for the Net class
			/// <summary>
			/// Auxiliary resources used in the calculation of the neural net's gradient. 
			/// </summary>
			std::vector<MLayerData<D>> layer_data_cache{};

			/// <summary>
			/// Gradient containers for all the layers.
			/// </summary>
			std::vector<MLayerGradient<D>> gradients{};

			/// <summary>
			/// Auxiliary resources used in the calculation of the neural net's value.
			/// </summary>
			InOutMData<D> value_cache{};

			/// <summary>
			/// Constructor allocating resources for the given number of layers.
			/// </summary>
			Context(const std::size_t layers_count) :
				layer_data_cache(layers_count + 1), // one more than the number of layers for technical
													// reasons (see the back-propagation method for more detail)
				gradients(layers_count){}

		public:
			/// <summary>
			/// Read-only access to the "out" of the value cache.
			/// </summary>
			const LazyVector<typename D::tensor_t>& get_out() const;

			/// <summary>
			/// Default constructor
			/// </summary>
			Context() = default;
		};

	private:
		static constexpr int MSG_PACK_VER = 1;

		std::vector<MLayerHandle<D>> _layers{};

		/// <summary>
		/// Evaluates gradient of the given <paramref name="cost_func"/> with respect
		/// to the given <paramref name="in_out"/> and <paramref name="reference"/>
		/// </summary>
		static void evaluate_cost_gradient_in_place(IMLayerExchangeData<typename D::tensor_t>& in_out,
			const IMLayerExchangeData<typename D::tensor_t>& reference, const CostFunction<typename D::tensor_t>& cost_func);

		/// <summary>
		/// Fills the given gradient containers with zero values.
		/// </summary>
		static void reset_gradients(std::vector<MLayerGradient<D>>& gradients);

		/// <summary>
		/// Calculates gradient of the neural net (with respect to its parameters) evaluated
		/// at the given <paramref name="input"/> - <paramref name="reference"/> pair and
		/// adds the result to the corresponding container in <paramref name="context"/>.
		/// </summary>
		void add_gradient(const LazyVector<typename D::tensor_t>& input,
			const LazyVector<typename D::tensor_t>& reference,
			const CostFunction<typename D::tensor_t>& cost_func, Context& context) const;

		/// <summary>
		/// Returns reference to a thread-local random number generator.
		/// </summary>
		static std::mt19937& ran_gen();

	public:

		/// <summary>
		/// Default constructor.
		/// </summary>
		MNet() = default;

		/// <summary>
		/// Appends a layer to the net.
		/// </summary>
		/// <typeparam name="L">Later type</typeparam>
		/// <typeparam name="...Types">Types of arguments required by a constructor of type "L"</typeparam>
		/// <param name="args">Actual arguments required by a constructor of type "L"</param>
		/// <returns>Output size of the appended layer</returns>
		template <template<class> class L, class... Types>
		Index4d append_layer(Types&&... args);

		/// <summary>
		/// Returns an instance of "context" that is supposed to be used to speed-up calculations.
		/// </summary>
		Context allocate_context() const;

		/// <summary>
		/// Allocates gradient containers for all the layers in the neural net.
		/// </summary>
		void allocate_gradients(std::vector<MLayerGradient<D>>& gradients, const bool fill_zero) const;

		/// <summary>
		/// Returns a collection of gradient containers allocated for the layers of the neural net.
		/// </summary>
		std::vector<MLayerGradient<D>> allocate_gradients(const bool fill_zero) const;

		/// <summary>
		/// Evaluated the neural net with respect to the given <paramref name="input"/>.
		///	The result can be accessed through <paramref name="context"/> parameter.
		/// </summary>
		void act(const IMLayerExchangeData<typename D::tensor_t>& input, Context& context) const;

		/// <summary>
		/// Returns value of the neural calculated at the given <paramref name="input"/>.
		/// </summary>
		LazyVector<typename D::tensor_t> act(const IMLayerExchangeData<typename D::tensor_t>& input) const;

		/// <summary>
		/// Calculates sum of gradients of the neural net (with respect to its parameters) evaluated
		/// for each of the <paramref name="input"/> - <paramref name="reference"/> pairs.
		/// The result is stored in the corresponding field of <paramref name="context"/>.
		/// </summary>
		void calc_gradient_sum(const LazyVector<LazyVector<typename D::tensor_t>>& input,
			const LazyVector<LazyVector<typename D::tensor_t>>& reference,
			const CostFunction<typename D::tensor_t>& cost_func, Context& context) const;

		/// <summary>
		/// Calculates and returns sum of gradients of the neural net (with respect to its parameters) evaluated
		/// for each of the <paramref name="input"/> - <paramref name="reference"/> pairs.
		/// </summary>
		std::vector<MLayerGradient<D>> calc_gradient_sum(const LazyVector<LazyVector<typename D::tensor_t>>& input,
			const LazyVector<LazyVector<typename D::tensor_t>>& reference,
			const CostFunction<typename D::tensor_t>& cost_func) const;

		/// <summary>
		/// Returns gradient of the neural net calculated with for the given
		/// <paramref name="input"/> - <paramref name="reference"/> pair.
		/// </summary>
		std::vector<MLayerGradient<D>> calc_gradient(const LazyVector<typename D::tensor_t>& input,
			const LazyVector<typename D::tensor_t>& reference, const CostFunction<typename D::tensor_t>& cost_func) const;

		/// <summary>
		/// Performs a single weight adjustment iteration based on the given <paramref name="input"/>,
		/// <paramref name="reference"/> and <paramref name="cost_func"/>
		/// </summary>
		void learn(const LazyVector<LazyVector<typename D::tensor_t>>& input,
			const LazyVector<LazyVector<typename D::tensor_t>>& reference,
			const CostFunction<typename D::tensor_t>& cost_func, const Real learning_rate, Context& context);

		/// <summary>
		/// Performs weights adjustment based on the given <paramref name="input"/>,
		/// <paramref name="reference"/> and <paramref name="cost_func"/>
		/// </summary>
		void learn(const LazyVector<LazyVector<typename D::tensor_t>>& input,
			const LazyVector<LazyVector<typename D::tensor_t>>& reference,
			const CostFunction<typename D::tensor_t>& cost_func, const Real learning_rate);

		/// <summary>
		/// Updates weights of all the layers according to the given gradient <paramref name="increments"/>
		/// and <paramref name="learning_rate"/>.
		/// </summary>
		void update(const std::vector<MLayerGradient<D>>& increments, const Real learning_rate);

		/// <summary>
		/// Returns number of multi-layers in the net.
		/// </summary>
		std::size_t layer_count() const;

		/// <summary>
		/// Size of the net's input.
		/// </summary>
		Index4d in_size() const;

		/// <summary>
		/// Size of the net's output.
		/// </summary>
		Index4d out_size() const;

		/// <summary>
		/// Resets random generator with the given seed.
		/// </summary>
		static void reset_random_generator(const unsigned seed);

		/// <summary>
		/// Resets random generator with the std::random_device generated seed.
		/// </summary>
		static void reset_random_generator();

		/// <summary>
		/// Equality operator.
		/// </summary>
		bool operator ==(const MNet& net) const;

		/// <summary>
		/// Inequality operator.
		/// </summary>
		bool operator !=(const MNet& net) const;

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const;

		/// <summary>
		/// Serializes current instance to the given file (in messagepack format)
		/// </summary>
		void save_to_file(const std::filesystem::path& file_name) const;

		/// <summary>
		/// Loads instance of the net from the given file (in messagepack format)
		/// </summary>
		static MNet load_from_file(const std::filesystem::path& file_name);
	};
}

#include "MNet.inl"