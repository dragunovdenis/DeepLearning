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
#include "../defs.h"
#include <msgpack.hpp>
#include "BasicCollection.h"
#include <filesystem>
#include "LinAlg3d.h"
#include "LinAlg2d.h"

namespace DeepLearning
{
	class PoolOperator;
	class Vector;
	class Matrix;

	/// <summary>
	/// Representation of rank 3 
	/// </summary>
	class Tensor : public BasicCollection
	{
		/// <summary>
		/// Number of layers (matrices) in the tensor
		/// </summary>
		std::size_t _layer_dim{};

		/// <summary>
		/// Number of rows in each layer
		/// </summary>
		std::size_t _row_dim{};

		/// <summary>
		/// Number of elements in each row (or columns in each layer)
		/// </summary>
		std::size_t _col_dim{};

		/// <summary>
		/// Releases allocated resources
		/// </summary>
		void free();

		/// <summary>
		/// Method to abandon resources (should be called when the resources are "moved")
		/// </summary>
		void abandon_resources();

		/// <summary>
		/// Converts given triplet of integer coordinates to a single index that can be used to access "data" array
		/// </summary>
		std::size_t coords_to_data_id(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const;

		/// <summary>
		/// Converts given index of an element in the "data" array to a triplet of layer, row and column indices of the same element
		/// </summary>
		Index3d data_id_to_index_3d(const long long data_id) const;

		/// <summary>
		/// Returns "true" if the given triplet of coordinates is valid to access "data" array
		/// </summary>
		bool check_bounds(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const;

	public:

		/// <summary>
		/// Return total number of elements in the tensor
		/// </summary>
		std::size_t size() const;

		/// <summary>
		/// Custom "packing" method
		/// </summary>
		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const
		{
			const auto proxy = to_stdvector();
			msgpack::type::make_define_array(_layer_dim, _row_dim, _col_dim, proxy).msgpack_pack(msgpack_pk);
		}

		/// <summary>
		/// Custom "unpacking" method
		/// </summary>
		void msgpack_unpack(msgpack::object const& msgpack_o);

		/// <summary>
		/// Default constructor
		/// </summary>
		Tensor() = default;

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="layer_dim">Layer dimension</param>
		/// <param name="row_dim">Row dimension in each layer</param>
		/// <param name="col_dim">Number of elements in each row</param>
		/// <param name="assign_zero">The tensor will be assigned with "0" if "true"</param>
		Tensor(const std::size_t layer_dim, const std::size_t row_dim,
			const std::size_t col_dim, const bool assign_zero = true);

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="size">Tensor size</param>
		/// <param name="assign_zero">The tensor will be assigned with "0" if "true"</param>
		Tensor(const Index3d& size, const bool assign_zero = true);

		/// <summary>
		/// Copy constructor
		/// </summary>
		Tensor(const Tensor& tensor);

		/// <summary>
		/// Move constructor
		/// </summary>
		Tensor(Vector&& vector) noexcept;

		/// <summary>
		/// Move constructor
		/// </summary>
		Tensor(Matrix&& matrix) noexcept;

		/// <summary>
		/// Move constructor
		/// </summary>
		Tensor(Tensor&& tensor) noexcept;

		/// <summary>
		/// Constructs dense tensor of the given dimension filled with
		/// uniformly distributed pseudo-random values from the given range
		/// </summary>
		Tensor(const std::size_t layer_dim, const std::size_t row_dim,
			const std::size_t col_dim, const Real range_begin, const Real range_end);

		/// <summary>
		/// Constructs dense tensor of the given dimension filled with
		/// uniformly distributed pseudo-random values from the given range
		/// </summary>
		Tensor(const Index3d& size, const Real range_begin, const Real range_end);

		/// <summary>
		/// Assignment operator
		/// </summary>
		Tensor& operator =(const Tensor& tensor);

		/// <summary>
		/// Move assignment operator for vector right-hand side operand
		/// </summary>
		Tensor& operator =(Vector&& vector) noexcept;

		/// <summary>
		/// Move assignment operator for matrix right-hand side operand
		/// </summary>
		Tensor& operator =(Matrix&& matrix) noexcept;

		/// <summary>
		/// Move assignment operator for tensor right-hand side operand
		/// </summary>
		Tensor& operator =(Tensor&& tensor) noexcept;

		/// <summary>
		/// Destructor
		/// </summary>
		~Tensor();

		/// <summary>
		/// Number of "layers"
		/// </summary>
		std::size_t layer_dim() const;

		/// <summary>
		/// Number of rows in each layer
		/// </summary>
		std::size_t row_dim() const;

		/// <summary>
		/// Number of columns in each layer (or number of elements in each layer row)
		/// </summary>
		std::size_t col_dim() const;

		/// <summary>
		/// Operator to access elements by three indices
		/// </summary>
		Real& operator ()(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id);

		/// <summary>
		/// Operator to access elements by three indices ("constant" version)
		/// </summary>
		const Real& operator ()(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const;

		/// <summary>
		/// Compound addition operator
		/// </summary>
		Tensor& operator +=(const Tensor & tensor);

		/// <summary>
		/// Compound subtraction operator
		/// </summary>
		Tensor& operator -=(const Tensor & tensor);

		/// <summary>
		/// Compound scalar multiplication operator
		/// </summary>
		Tensor& operator *=(const Real & scalar);

		/// <summary>
		/// "Equal to" operator
		/// </summary>
		bool operator ==(const Tensor & tensor) const;

		/// <summary>
		/// "Not equal to" operator
		/// </summary>
		bool operator !=(const Tensor& tensor) const;

		/// <summary>
		/// Returns "sizes" of the tensor in all the 3 directions
		/// </summary>
		Index3d size_3d() const;

		/// <summary>
		/// Changes size of the tensor without modifying of the underlying data container
		/// </summary>
		/// <param name="new_shape">New sizes of the tensor. Must be consistent with the
		/// length of the underlying data container, otherwise exception will be thrown</param>
		Tensor& reshape(const Index3d& new_shape);

		/// <summary>
		/// Returns total size of convolution result
		/// </summary>
		/// <param name="tensor_size">Size of the tensor the convolution is to be applied to</param>
		/// <param name="kernel_size">Size of the convolution kernel</param>
		/// <param name="paddings">Sizes of zero paddings of the tensor</param>
		/// <param name="strides">Sizes of strides to be used</param>
		static Index3d calc_conv_res_size(const Index3d& tensor_size, const Index3d& kernel_size, const Index3d& paddings, const Index3d& strides);

		/// <summary>
		/// Convolution with another tensor
		/// </summary>
		/// <param name="result_handle">Handle of the memory allocated by the caller to store the result of the convolution.
		/// It is supposed that the size of the handle is equal to the product of sizes in three dimensions returned by method `calc_conv_res_size()`.
		/// Otherwise an exception will be thrown.</param>
		/// <param name="kernel">Convolution kernel</param>
		/// <param name="paddings">Paddings in 3-dimensional index space</param>
		/// <param name="strides">Strides in 3-dimensional index space</param>
		/// <returns>Size of the convolution result. To be used to interpret the memory pointed by the handle parameter.</returns>
		Index3d convolve(RealMemHandle result_handle, const Tensor& kernel, const Index3d& paddings, const Index3d& strides) const;

		/// <summary>
		/// Convolution with another tensor
		/// </summary>
		/// <param name="kernel">Convolution kernel</param>
		/// <param name="paddings">Paddings in 3-dimensional index space</param>
		/// <param name="strides">Strides in 3-dimensional index space</param>
		/// <returns>Result of the convolution</returns>
		Tensor convolve(const Tensor& kernel, const Index3d& paddings = Index3d{ 0, 0, 0 },
											  const Index3d& strides = Index3d{ 1, 1, 1 }) const;

		/// <summary>
		/// Computes gradient of some scalar function F (depending on the result of some convolution)
		/// with respect to the convolution kernel K: dF/dK
		/// and to the input tensor of the convolution I : dF/dI
		/// </summary>
		/// <param name="conv_res_grad">Gradient of the function F with respect to the result of the convolution R: dF/dR</param>
		/// <param name="kernel">The convolution kernel</param>
		/// <param name="paddings">Paddings used for computing the convolution</param>
		/// <param name="strides">Strides used for computing the convolution</param>
		/// <returns>Tuple of tensors dF/dK, dF/dI in the exact same order </returns>
		std::tuple<Tensor, Tensor> convolution_gradient(const Tensor& conv_res_grad, const Tensor& kernel, const Index3d& paddings,
			const Index3d& strides) const;

		/// <summary>
		/// Computes gradient of some scalar function F (depending on the result of some convolution)
		/// with respect to the convolution kernel K: dF/dK
		/// and to the input tensor of the convolution I : dF/dI
		/// </summary>
		/// <param name="conv_res_grad">Handle of the memory containing gradient of the function F
		/// with respect to the result of the convolution R: dF/dR. </param>
		/// <param name="kernel">The convolution kernel</param>
		/// <param name="paddings">Paddings used for computing the convolution</param>
		/// <param name="strides">Strides used for computing the convolution</param>
		/// <returns>Tuple of tensors dF/dK, dF/dI in the exact same order </returns>
		std::tuple<Tensor, Tensor> convolution_gradient(const RealMemHandleConst& conv_res_grad, const Tensor& kernel, const Index3d& paddings,
			const Index3d& strides) const;

		/// <summary>
		/// More general implementation of the convolution operation, that can perform pooling operations
		/// </summary>
		/// <param name="pool_operator">Instance of the pool operator to be applied (generalization of the convolution kernel)</param>
		/// <param name="kernel_size">Size of the "window" for the pooling agent operate</param>
		/// <param name="paddings">Zero paddings (will be applied to the base tensor)</param>
		/// <param name="strides">Strides defining movement of the "window"</param>
		Tensor pool(const PoolOperator& pool_operator, const Index3d& paddings = Index3d{ 0, 0, 0 },
			const Index3d& strides = Index3d{ 1, 1, 1 }) const;

		/// <summary>
		/// More general implementation of the convolution operation, that can perform pooling operations
		/// </summary>
		/// <param name="result_handle">Handle of the memory allocated by the caller to store the result of the convolution.
		/// It is supposed that the size of the handle is equal to the product of sizes in three dimensions returned by method `calc_conv_res_size()`.
		/// Otherwise an exception will be thrown.</param>
		/// <param name="pool_operator">Instance of the pool operator to be applied (generalization of the convolution kernel)</param>
		/// <param name="kernel_size">Size of the "window" for the pooling agent operate</param>
		/// <param name="paddings">Zero paddings (will be applied to the base tensor)</param>
		/// <param name="strides">Strides defining movement of the "window"</param>
		/// <returns>Size of the convolution result. To be used to interpret the memory pointed by the handle parameter.</returns>
		Index3d pool(RealMemHandle result_handle, const PoolOperator& pool_operator, const Index3d& paddings = Index3d{ 0, 0, 0 },
			const Index3d& strides = Index3d{ 1, 1, 1 }) const;

		/// <summary>
		/// Returns gradient of some function F (depending on the pooling result) with respect to the pooling input tensor I: dF/dI
		/// </summary>
		/// <param name="pool_operator">Instance of the pool operator to be applied (generalization of the convolution kernel)</param>
		/// <param name="pool_res_grad">Gradient of the function F with respect to the result of the pooling operation R: dF/dR</param>
		/// <param name="kernel_size">Size of the "window" for the pooling agent operate</param>
		/// <param name="paddings">Zero paddings (will be applied to the base tensor)</param>
		/// <param name="strides">Strides defining movement of the "window"</param>
		Tensor pool_input_gradient(const Tensor& pool_res_grad, const PoolOperator& pool_operator, const Index3d& paddings,
			const Index3d& strides) const;

		/// <summary>
		/// Returns gradient of some function F (depending on the pooling result) with respect to the pooling input tensor I: dF/dI
		/// </summary>
		/// <param name="pool_operator">Instance of the pool operator to be applied (generalization of the convolution kernel)</param>
		/// <param name="pool_res_grad">Handle of the memory containing gradient of the function F with respect to the result of the pooling operation R: dF/dR</param>
		/// <param name="kernel_size">Size of the "window" for the pooling agent operate</param>
		/// <param name="paddings">Zero paddings (will be applied to the base tensor)</param>
		/// <param name="strides">Strides defining movement of the "window"</param>
		Tensor pool_input_gradient(const RealMemHandleConst& pool_res_grad, const PoolOperator& pool_operator, const Index3d& paddings,
			const Index3d& strides) const;

		/// <summary>
		/// Min-max specialized version of pool algorithm, which is more optimal than the "pool" method above with the corresponding pool-operator.
		/// Simplified version, with zero paddings and strides equal to the window dimensions.
		/// Returns a tuple containing the result of pooling and an vector of indices which is a mapping from the indices of flattened result
		/// of the pooling to the flattened indices of input elements that have been pooled pooling.
		/// The mapping allows to simplify a back-propagation procedure.
		/// </summary>
		/// <param name="window">Operation window size</param>
		/// <param name="max">If "true" the method implements "max pulling" otherwise -- "min pulling";</param>
		/// <returns></returns>
		std::tuple<Tensor, std::vector<std::size_t>> min_max_pool_2d(const Index2d& window_size, const bool max) const;

		/// <summary>
		/// Returns gradient of some function F with respect to the min/max 2d pool input tensor I: dF/dI
		/// </summary>
		/// <param name="pool_res_gradient">Gradient of the function F with respect to the min/max 2d pool output tensor O: dF/dO</param>
		/// <param name="out_to_in_mapping">Min/max 2d pool output to input flattened index mapping (the second item in the tuple returned by min_max_pool_2d)</param>
		Tensor min_max_pool_2d_input_gradient(const Tensor& pool_res_gradient, const std::vector<std::size_t>& out_to_in_mapping) const;

		/// <summary>
		/// Returns read-only memory handle to the layer with given index
		/// </summary>
		/// <param name="layer_id">Index of a layer</param>
		RealMemHandleConst get_layer_handle(const std::size_t& layer_id) const;

		/// <summary>
		/// Returns memory handle to the layer with given index
		/// </summary>
		/// <param name="layer_id">Index of a layer</param>
		RealMemHandle get_layer_handle(const std::size_t& layer_id);

		/// <summary>
		/// Writes layer with the given id to a text file on disk in a form of rectangular table
		/// </summary>
		/// <param name="layer_id">Layer identifier</param>
		/// <param name="filename">Full file path to write to</param>
		void log_layer(const std::size_t& layer_id, const std::filesystem::path& filename) const;

		/// <summary>
		/// Logs entire tensor to a text files into the specified directory (each layer to separate file)
		/// </summary>
		/// <param name="directory">The directory to log to. Must exist when the method is called</param>
		/// <param name="base_log_name">Basic name of the layer-wise log files (no extension)</param>
		void log(const std::filesystem::path& directory, const std::filesystem::path& base_log_name) const;
	};

	/// <summary>
	/// Addition operator
	/// </summary>
	Tensor operator +(const Tensor& tensor1, const Tensor& tensor2);

	/// <summary>
	/// Subtraction operator
	/// </summary>
	Tensor operator -(const Tensor& tensor1, const Tensor& tensor2);

	/// <summary>
	/// Multiplication by a scalar from the right
	/// </summary>
	Tensor operator *(const Tensor& tensor, const Real& scalar);

	/// <summary>
	/// Multiplication by a scalar from the left
	/// </summary>
	Tensor operator *(const Real& scalar, const Tensor& tensor);

	/// <summary>
	/// Compound addition operator for a 4-dimensional tensor (i.e. vector of 3d tensosrs)
	/// </summary>
	/// <param name="op1">Operand to add to</param>
	/// <param name="op2">Operand to be added to</param>
	/// <returns>Reference to the modified operand</returns>
	std::vector<Tensor>& operator +=(std::vector<Tensor>& op1, const std::vector<Tensor>& op2);

	/// <summary>
	/// Compound scalar multiplication operator for a 4-dimensional tensor (i.e. vector of 3d tensosrs)
	/// </summary>
	/// <param name="op1">Operand to add to</param>
	/// <param name="scalar">Scalar to multiply by</param>
	/// <returns>Reference to the modified operand</returns>
	std::vector<Tensor>& operator *=(std::vector<Tensor>& op1, const Real& scalar);

}
