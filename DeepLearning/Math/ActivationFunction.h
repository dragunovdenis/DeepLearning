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
#include<msgpack.hpp>
#include <memory>
#include "ActivationFunctionId.h"

namespace DeepLearning
{
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
		T operator ()(const T& input) const;

		/// <summary>
		/// Calculates functional value for all the elements of the given input collection in place
		/// </summary>
		/// <param name="in_out">Collection of input elements that, after method returns,
		/// will contain the result of function evaluation</param>
		virtual void func_in_place(T& in_out) const = 0;

		/// <summary>
		/// Calculates function and auxiliary data needed to calculate gradient with respect to the input
		/// </summary>
		std::tuple<T, T> func_and_aux(const T& input) const;

		/// <summary>
		/// Calculates function and auxiliary data needed to calculate gradient with respect to the input
		/// Function value is calculated "in-place"
		/// </summary>
		/// <param name="in_out">Collection of input elements that, after method returns,
		/// will contain the result of function evaluation</param>
		/// <param name="aux">Place-holder for the auxiliary data</param>
		virtual void func_and_aux_in_place(T& in_out, T& aux) const = 0;

		/// <summary>
		/// Calculates gradient with respect to the function's input 
		/// </summary>
		/// <param name="out_grad">Gradient with respect to the function's output</param>
		/// <param name="aux_data">Auxiliary data calculated by function "func_and_aux"</param>
		T get_in_grad(const typename T::Base& out_grad, const T& aux_data) const;

		/// <summary>
		/// Calculates gradient with respect to the function's input 
		/// </summary>
		/// <param name="out_grad">Gradient with respect to the function's output</param>
		/// <param name="aux_data">Auxiliary data calculated by function "func_and_aux"</param>
		/// <param name="out">Place-holder for the result of calculation</param>
		virtual void calc_in_grad(const typename T::Base& out_grad, const T& aux_data, T& out) const;

		/// <summary>
		/// Calculates gradient with respect to the function's input and adds it to <paramref name="out_res"/>
		/// </summary>
		/// <param name="out_grad">Gradient with respect to the function's output</param>
		/// <param name="aux_data">Auxiliary data calculated by function "func_and_aux"</param>
		/// <param name="out_res">Collection to add the results of calculation to.</param>
		virtual void add_in_grad(const typename T::Base& out_grad, const T& aux_data, T& out_res) const;

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
		std::shared_ptr<AFunction<T>> _func{};
	public:
		/// <summary>
		/// Constructor
		/// </summary>
		ActivationWrapper(const ActivationFunctionId id);

		/// <summary>
		/// Operator to access the reference to "wrapped" function
		/// </summary>
		const AFunction<T>& operator()() const;

		/// <summary>
		/// Factory method.
		/// </summary>
		static std::shared_ptr<AFunction<T>> construct(const ActivationFunctionId id);
	};

	/// <summary>
	/// Activation function
	/// </summary>
	template <class T>
	class ActivationFunction : public AFunction<T>
	{
		const ActivationFunctionId _id;
	public:

		/// <summary>
		/// Constructor
		/// </summary>
		ActivationFunction(const ActivationFunctionId id);

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_in_place(T& in_out) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_and_aux_in_place(T& in_out, T& aux) const override;
	};

	/// <summary>
	/// Soft-max activation function
	/// </summary>
	template <class T>
	class SoftMaxActivationFunction : public AFunction<T>
	{
	public:

		/// <summary>
		/// Default constructor
		/// </summary>
		SoftMaxActivationFunction() = default;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_in_place(T& in_out) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_and_aux_in_place(T& in_out, T& aux) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void calc_in_grad(const typename T::Base& out_grad, const T& aux_data, T& result) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void add_in_grad(const typename T::Base& out_grad, const T& aux_data, T& out_res) const override;
	};

	/// <summary>
	/// ReLu activation function
	/// </summary>
	template <class T>
	class ReLuActivationFunction : public AFunction<T>
	{
	public:

		/// <summary>
		/// Default constructor
		/// </summary>
		ReLuActivationFunction() = default;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_in_place(T& in_out) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_and_aux_in_place(T& in_out, T& aux) const override;
	};

	/// <summary>
	/// Sigmoid activation function
	/// </summary>
	template <class T>
	class SigmoidActivationFunction : public AFunction<T>
	{
	public:

		/// <summary>
		/// Default constructor
		/// </summary>
		SigmoidActivationFunction() = default;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_in_place(T& in_out) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_and_aux_in_place(T& in_out, T& aux) const override;
	};

	/// <summary>
	/// "Tangent hyperbolic" activation function
	/// </summary>
	template <class T>
	class TanhActivationFunction : public AFunction<T>
	{
	public:

		/// <summary>
		/// Default constructor
		/// </summary>
		TanhActivationFunction() = default;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_in_place(T& in_out) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		void func_and_aux_in_place(T& in_out, T& aux) const override;
	};
}

#include "ActivationFunction.inl"

MSGPACK_ADD_ENUM(DeepLearning::ActivationFunctionId)
