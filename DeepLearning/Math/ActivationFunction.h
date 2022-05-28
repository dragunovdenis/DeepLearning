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
#include "DiffFunc.h"
#include<msgpack.hpp>
#include <memory>

namespace DeepLearning
{
	class BasicCollection;

	/// <summary>
	/// Identifiers of different activation functions
	/// </summary>
	enum class ActivationFunctionId: unsigned int {
		UNKNOWN = 0,
		SIGMOID = 1, //Sigmoid activation function
		TANH = 2, //Hyperbolic tangent activation function
		RELU = 3, // rectified linear activation (unit)
		SOFTMAX = 4,//soft-max function
	};

	/// <summary>
	/// Returns string representation of the given activation function type identifier
	/// </summary>
	std::string to_string(const ActivationFunctionId& activation_type_id);

	/// <summary>
	/// Parses given string to the activation function identifier
	/// </summary>
	ActivationFunctionId parse_activation_type(const std::string& str);

	/// <summary>
	/// General interface of an activation function
	/// </summary>
	template <class T>
	class AFunction
	{
	public:

		/// <summary>
		/// The function
		/// </summary>
		virtual T operator ()(const T& input) const = 0;

		/// <summary>
		/// Calculates function and auxiliary data needed to calculate gradient with respect to the input
		/// </summary>
		virtual std::tuple<T, T> func_and_aux(const T& input) const = 0;

		/// <summary>
		/// Calculates gradient with respect to the function's input 
		/// </summary>
		/// <param name="out_grad">Gradient with respect to the function's output</param>
		/// <param name="aux_data">Auxiliary data calculated by function "func_and_aux"</param>
		virtual T calc_input_gradient(const BasicCollection& out_grad, const T& aux_data) const = 0;

		/// <summary>
		/// Virtual destructor to ensure proper releasing of the resources of descending classes
		/// </summary>
		virtual ~AFunction() {}
	};

	/// <summary>
	/// Factory for instantiating activation functions by their identifiers
	/// </summary>
	template <class T>
	class ActivationWrapper
	{
		std::unique_ptr<AFunction<T>> _func{};
	public:
		/// <summary>
		/// Constructor
		/// </summary>
		ActivationWrapper(const ActivationFunctionId id);

		/// <summary>
		/// Operator to access the reference to "wrapped" function
		/// </summary>
		const AFunction<T>& operator()() const;
	};

	/// <summary>
	/// Activation function
	/// </summary>
	template <class T>
	class ActivationFuncion : public AFunction<T>
	{
		std::unique_ptr<DiffFunc> _func{};

	public:

		/// <summary>
		/// Constructor
		/// </summary>
		ActivationFuncion(const ActivationFunctionId id);

		/// <summary>
		/// The function
		/// </summary>
		virtual T operator ()(const T& input) const override;

		/// <summary>
		/// Calculates function and auxiliary data needed to calculate gradient with respect to the input
		/// </summary>
		virtual std::tuple<T, T> func_and_aux(const T& input) const override;

		/// <summary>
		/// Calculates gradient with respect to the function's input 
		/// </summary>
		/// <param name="out_grad">Gradient with respect to the function's output</param>
		/// <param name="aux_data">Auxiliary data calculated by function "func_and_aux"</param>
		virtual T calc_input_gradient(const BasicCollection& out_grad, const T& aux_data) const override;
	};

	/// <summary>
	/// Sort-max activation function
	/// </summary>
	template <class T>
	class SoftMaxActivationFuncion : public AFunction<T>
	{
		/// <summary>
		/// Calculates a collection containing exponents of the normalized input elements
		/// </summary>
		T calc_aux_data(const T& input) const;

	public:

		/// <summary>
		/// Default constructor
		/// </summary>
		SoftMaxActivationFuncion() = default;

		/// <summary>
		/// The function
		/// </summary>
		virtual T operator ()(const T& input) const override;

		/// <summary>
		/// Calculates function and auxiliary data needed to calculate gradient with respect to the input
		/// </summary>
		virtual std::tuple<T, T> func_and_aux(const T& input) const override;

		/// <summary>
		/// Calculates gradient with respect to the function's input 
		/// </summary>
		/// <param name="out_grad">Gradient with respect to the function's output</param>
		/// <param name="aux_data">Auxiliary data calculated by function "func_and_aux"</param>
		virtual T calc_input_gradient(const BasicCollection& out_grad, const T& aux_data) const override;
	};
}

MSGPACK_ADD_ENUM(DeepLearning::ActivationFunctionId)
