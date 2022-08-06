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
#include "../Utilities.h"
#include "../CudaBridge.h"
#include "Functions.h"

namespace DeepLearning
{
	class BasicCollection;
	class BasicCudaCollection;

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
		/// Calculates functional value for all the elements of the given input collection in place
		/// </summary>
		/// <param name="in_out">Collection of input elements that, after method returns,
		/// will contain the result of function evaluation</param>
		virtual void func_in_place(T& in_out) const = 0;

		/// <summary>
		/// Calculates function and auxiliary data needed to calculate gradient with respect to the input
		/// </summary>
		virtual std::tuple<T, T> func_and_aux(const T& input) const = 0;

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
		virtual T calc_input_gradient(const typename T::Base& out_grad, const T& aux_data) const = 0;

		/// <summary>
		/// Calculates gradient with respect to the function's input 
		/// </summary>
		/// <param name="out_grad">Gradient with respect to the function's output</param>
		/// <param name="aux_data">Auxiliary data calculated by function "func_and_aux"</param>
		/// <param name="out">Place-holder for the result of calculation</param>
		virtual void calc_input_gradient(const typename T::Base& out_grad, const T& aux_data, T& out) const = 0;

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
	/// Helper methods used for activation function evaluation
	/// </summary>
	namespace ActivationFunctionHelper
	{
		/// <summary>
		/// A factory method: the only "legal" way to instantiate an activation function via its identifier
		/// </summary>
		template <class F>
		CUDA_CALLABLE F make(const ActivationFunctionId id)
		{
			switch (id)
			{
			case ActivationFunctionId::SIGMOID: return [](const auto& x) { return Func::sigmoid(x); };
			case ActivationFunctionId::TANH: return [](const auto& x) { return tanh(x); };
			case ActivationFunctionId::RELU: return [](const auto& x) { return  x < Real(0) ? Real(0) : x; };
			default: return [](const auto& x) { return decltype(x)(std::numeric_limits<Real>::signaling_NaN()); };
			}
		}

		/// <summary>
		/// Evaluates given function at each element of the given collection and stores the result "in place"
		/// </summary>
		void evaluate_in_place(BasicCudaCollection& collection, const ActivationFunctionId id);

		/// <summary>
		/// Evaluates given function at each element of the given collection and stores the result "in place"
		/// </summary>
		void evaluate_in_place(BasicCollection& collection, const ActivationFunctionId id);

		/// <summary>
		/// Evaluates given function and its derivative at each element of the given "collection_func"
		/// Stores the function value to the "collection_func" whereas the derivative value is stored to the "collection_deriv"
		/// </summary>
		void evaluate_in_place(BasicCollection& collection_func, BasicCollection& collection_deriv, const ActivationFunctionId id);

		/// <summary>
		/// Evaluates given function and its derivative at each element of the given "collection_func"
		/// Stores the function value to the "collection_func" whereas the derivative value is stored to the "collection_deriv"
		/// </summary>
		void evaluate_in_place(BasicCudaCollection& collection_func, BasicCudaCollection& collection_deriv, const ActivationFunctionId id);

		/// <summary>
		/// Subtracts maximal element in the given collection from each element of the collection,
		/// evaluated exponent of each element and sore the result to the given collection
		/// </summary>
		void normalize_and_evaluate_exponent_in_place(BasicCollection& collection);

		/// <summary>
		/// Subtracts maximal element in the given collection from each element of the collection,
		/// evaluated exponent of each element and sore the result to the given collection
		/// </summary>
		void normalize_and_evaluate_exponent_in_place(BasicCudaCollection& collection);

		/// <summary>
		/// Evaluates gradient of the soft-max function with respect to its input
		/// </summary>
		/// <param name="input_exp">Collection containing exponents of the soft-max input</param>
		/// <param name="out_grad">Gradient with respect to the output of soft-max</param>
		/// <param name="result">Placeholder for the calculated gradient. Should be allocated by the caller;
		/// Must be initialized with a copy of "input_exp" by the caller</param>
		void evaluate_softmax_input_grad( const BasicCollection& input_exp, const BasicCollection& out_grad, BasicCollection& result);

		/// <summary>
		/// Evaluates gradient of the soft-max function with respect to its input
		/// </summary>
		/// <param name="input_exp">Collection containing exponents of the soft-max input</param>
		/// <param name="result">Placeholder for the calculated gradient. Should be allocated by the caller
		/// Must be initialized with a copy of "input_exp" by the caller</param>
		void evaluate_softmax_input_grad(const BasicCudaCollection& input_exp, const BasicCudaCollection& out_grad, BasicCudaCollection& result);
	}

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
		/// The function
		/// </summary>
		virtual T operator ()(const T& input) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		virtual void func_in_place(T& in_out) const override;

		/// <summary>
		/// Calculates function and auxiliary data needed to calculate gradient with respect to the input
		/// </summary>
		virtual std::tuple<T, T> func_and_aux(const T& input) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		virtual void func_and_aux_in_place(T& in_out, T& aux) const override;

		/// <summary>
		/// Calculates gradient with respect to the function's input 
		/// </summary>
		/// <param name="out_grad">Gradient with respect to the function's output</param>
		/// <param name="aux_data">Auxiliary data calculated by function "func_and_aux"</param>
		virtual T calc_input_gradient(const typename T::Base& out_grad, const T& aux_data) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		virtual void calc_input_gradient(const typename T::Base& out_grad, const T& aux_data, T& result) const override;
	};

	/// <summary>
	/// Sort-max activation function
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
		/// The function
		/// </summary>
		virtual T operator ()(const T& input) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		virtual void func_in_place(T& in_out) const override;

		/// <summary>
		/// Calculates function and auxiliary data needed to calculate gradient with respect to the input
		/// </summary>
		virtual std::tuple<T, T> func_and_aux(const T& input) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		virtual void func_and_aux_in_place(T& in_out, T& aux) const override;

		/// <summary>
		/// Calculates gradient with respect to the function's input 
		/// </summary>
		/// <param name="out_grad">Gradient with respect to the function's output</param>
		/// <param name="aux_data">Auxiliary data calculated by function "func_and_aux"</param>
		virtual T calc_input_gradient(const typename T::Base& out_grad, const T& aux_data) const override;

		/// <summary>
		/// See the summary of the corresponding method in the base class
		/// </summary>
		virtual void calc_input_gradient(const typename T::Base& out_grad, const T& aux_data, T& result) const override;
	};
}

MSGPACK_ADD_ENUM(DeepLearning::ActivationFunctionId)
