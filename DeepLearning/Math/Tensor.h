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
#include "LinAlg3d.h"

namespace DeepLearning
{
	class PoolOperator;

	/// <summary>
	/// Representation of rank 3 
	/// </summary>
	class Tensor : public BasicCollection
	{
		Real* _data{};

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

		template <typename Packer>
		void msgpack_pack(Packer& msgpack_pk) const
		{
			const auto proxy = std::vector<Real>(begin(), end());
			msgpack::type::make_define_array(_layer_dim, _row_dim, _col_dim, proxy).msgpack_pack(msgpack_pk);
		}

		void msgpack_unpack(msgpack::object const& msgpack_o)
		{
			std::vector<Real> proxy;
			msgpack::type::make_define_array(_layer_dim, _row_dim, _col_dim, proxy).msgpack_unpack(msgpack_o);
			_data = reinterpret_cast<Real*>(std::malloc(size() * sizeof(Real)));
			std::copy(proxy.begin(), proxy.end(), begin());
		}

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
		/// Copy constructor
		/// </summary>
		Tensor(const Tensor& tensor);

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
		/// Assignment operator
		/// </summary>
		Tensor& operator =(const Tensor& tensor);

		/// <summary>
		/// Destructor
		/// </summary>
		~Tensor();

		/// <summary>
		/// Pointer to the first element of the tensor
		/// </summary>
		Real* begin();

		/// <summary>
		/// Pointer to the first element of the tensor (constant version)
		/// </summary>
		const Real* begin() const;

		/// <summary>
		/// Pointer to the "behind last" element of the tensor
		/// </summary>
		Real* end();

		/// <summary>
		/// Pointer to the "behind last" element of the tensor (constant version)
		/// </summary>
		const Real* end() const;

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
		/// More general implementation of the convolution operation, that can perform pooling operations
		/// </summary>
		/// <param name="pool_operator">Instance of the pool operator to be applied (generalization of the convolution kernel)</param>
		/// <param name="kernel_size">Size of the "window" for the pooling agent operate</param>
		/// <param name="paddings">Zero paddings (will be applied to the base tensor)</param>
		/// <param name="strides">Strides defining movement of the "window"</param>
		Tensor pool(const PoolOperator& pool_operator, const Index3d& paddings = Index3d{ 0, 0, 0 },
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

}
