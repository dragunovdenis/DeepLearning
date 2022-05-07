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

#include "Tensor.h"
#include "../Utilities.h"
#include <exception>
#include "../IndexIterator.h"
#include "PoolOperator.h"
#include "Vector.h"
#include "Matrix.h"

namespace DeepLearning
{
	Tensor::Tensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const bool assign_zero)
		: _layer_dim(layer_dim), _row_dim(row_dim), _col_dim(col_dim)
	{
		_data = reinterpret_cast<Real*>(std::malloc(size() * sizeof(Real)));

		if (assign_zero)
			std::fill(begin(), end(), Real(0));
	}

	Tensor::Tensor(const Index3d& size, const bool assign_zero)
		: Tensor(size.x, size.y, size.z, assign_zero)
	{}

	Tensor::Tensor(const Tensor& tensor)
		: Tensor(tensor._layer_dim, tensor._row_dim, tensor._col_dim, false)
	{
		std::copy(tensor.begin(), tensor.end(), begin());
	}

	void Tensor::abandon_resources()
	{
		_data = nullptr;
		_layer_dim = 0;
		_row_dim = 0;
		_col_dim = 0;
	}

	Tensor::Tensor(Tensor&& tensor) noexcept
		: _data(tensor._data), _layer_dim(tensor._layer_dim),
		_row_dim(tensor._row_dim), _col_dim(tensor._col_dim)
	{
		tensor.abandon_resources();
	}

	Tensor::Tensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const Real range_begin, const Real range_end)
		: Tensor(layer_dim, row_dim, col_dim, false)
	{
		Utils::fill_with_random_values(begin(), end(), range_begin, range_end);
	}

	Tensor& Tensor::operator =(const Tensor& tensor)
	{
		if (size() != tensor.size())
		{
			free();
			_data = reinterpret_cast<Real*>(std::malloc(tensor.size() * sizeof(Real)));
		}

		_layer_dim = tensor._layer_dim;
		_row_dim = tensor._row_dim;
		_col_dim = tensor._col_dim;

		std::copy(tensor.begin(), tensor.end(), begin());

		return *this;
	}

	/// <summary>
	/// Move constructor
	/// </summary>
	Tensor::Tensor(Vector&& vector) noexcept : _layer_dim(1ull),
		_row_dim(1ull), _col_dim(vector.dim()), _data(vector.begin())
	{
		vector.abandon_resources();
	}

	/// <summary>
	/// Move constructor
	/// </summary>
	Tensor::Tensor(Matrix&& matrix) noexcept : _layer_dim(1ull),
		_row_dim(matrix.row_dim()), _col_dim(matrix.col_dim()), _data(matrix.begin())
	{
		matrix.abandon_resources();
	}

	Tensor& Tensor::operator =(Vector&& vector) noexcept
	{
		free();

		_layer_dim = 1ull;
		_row_dim = 1ull;
		_col_dim = vector.dim();

		_data = vector.begin();
		vector.abandon_resources();

		return *this;
	}

	Tensor& Tensor::operator =(Matrix&& matrix) noexcept
	{
		free();

		_layer_dim = 1ull;
		_row_dim = matrix.row_dim();
		_col_dim = matrix.col_dim();

		_data = matrix.begin();
		matrix.abandon_resources();

		return *this;
	}

	Tensor& Tensor::operator =(Tensor&& tensor) noexcept
	{
		free();

		_layer_dim = tensor.layer_dim();
		_row_dim = tensor.row_dim();
		_col_dim = tensor.col_dim();

		_data = tensor.begin();
		tensor.abandon_resources();

		return *this;
	}

	Tensor::~Tensor()
	{
		free();
	}

	void Tensor::free()
	{
		if (_data != nullptr)
		{
			delete[] _data;
			_data = nullptr;
		}

		_layer_dim = 0;
		_row_dim = 0;
		_col_dim = 0;
	}

	std::size_t Tensor::size() const
	{
		return _layer_dim * _row_dim * _col_dim;
	}

	Real* Tensor::begin()
	{
		return _data;
	}

	const Real* Tensor::begin() const
	{
		return _data;
	}

	Real* Tensor::end()
	{
		return _data + size();
	}

	const Real* Tensor::end() const
	{
		return _data + size();
	}

	std::size_t Tensor::layer_dim() const
	{
		return _layer_dim;
	}

	std::size_t Tensor::row_dim() const
	{
		return _row_dim;
	}

	std::size_t Tensor::col_dim() const
	{
		return _col_dim;
	}

	std::size_t Tensor::coords_to_data_id(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const
	{
		return _col_dim * (layer_id * _row_dim + row_id) + col_id;
	}

	/// <summary>
	/// Converts given index of an element in the "data" array to a triplet of layer, row and column indices of the same element
	/// </summary>
    /// <param name="data_id">Index of an element in the "data" array</param>
	/// <param name="tensor_size">3d size of the tensor</param>
	inline Index3d data_id_to_index_3d_internal(const long long data_id, const Index3d& tensor_size)
	{
		const auto temp = std::div(data_id, static_cast<long long>(tensor_size.z));
		const auto col_id = temp.rem;
		const auto temp1 = std::div(temp.quot, static_cast<long long>(tensor_size.y));
		const auto row_id = temp1.rem;
		const auto layer_id = temp1.quot;

		return { layer_id, row_id, col_id };
	}

	Index3d Tensor::data_id_to_index_3d(const long long data_id) const
	{
		return data_id_to_index_3d_internal(data_id, size_3d());
	}

	bool Tensor::check_bounds(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const
	{
		return layer_id < _layer_dim && row_id < _row_dim && col_id < _col_dim;
	}

	Real& Tensor::operator ()(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, row_id, col_id))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return _data[coords_to_data_id(layer_id, row_id, col_id)];
	}

	const Real& Tensor::operator ()(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, row_id, col_id))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return _data[coords_to_data_id(layer_id, row_id, col_id)];
	}

	Tensor& Tensor::operator +=(const Tensor& tensor)
	{
		if (_layer_dim != tensor._layer_dim ||
			_row_dim != tensor._row_dim ||
			_col_dim != tensor._col_dim)
			throw std::exception("Invalid input");

		add(tensor);
		return *this;
	}

	Tensor& Tensor::operator -=(const Tensor& tensor)
	{
		if (_layer_dim != tensor._layer_dim ||
			_row_dim != tensor._row_dim ||
			_col_dim != tensor._col_dim)
			throw std::exception("Invalid input");

		sub(tensor);
		return *this;
	}

	Tensor& Tensor::operator *=(const Real& scalar)
	{
		mul(scalar);
		return *this;
	}

	Tensor operator +(const Tensor& tensor1, const Tensor& tensor2)
	{
		auto result = tensor1;
		result += tensor2;

		return result;
	}

	Tensor operator -(const Tensor& tensor1, const Tensor& tensor2)
	{
		auto result = tensor1;
		result -= tensor2;

		return result;
	}

	Tensor operator *(const Tensor& tensor, const Real& scalar)
	{
		auto result = tensor;
		result *= scalar;

		return result;
	}

	Tensor operator *(const Real& scalar, const Tensor& tensor)
	{
		return tensor * scalar;
	}

	bool Tensor::operator ==(const Tensor& tensor) const
	{
		return _layer_dim == tensor._layer_dim &&
			_row_dim == tensor._row_dim &&
			_col_dim == tensor._col_dim &&
			std::all_of(IndexIterator(0), IndexIterator(static_cast<int>(size())),
				[&](const auto& id) { return _data[id] == tensor._data[id]; });
	}

	bool Tensor::operator !=(const Tensor& tensor) const
	{
		return !(*this == tensor);
	}

	Index3d Tensor::size_3d() const
	{
		return Index3d{ static_cast<long long>(_layer_dim),
			            static_cast<long long>(_row_dim),
			            static_cast<long long>(_col_dim) };
	}

	Tensor& Tensor::reshape(const Index3d& new_shape)
	{
		if (size() != new_shape.x * new_shape.y * new_shape.z)
			throw std::exception("Invalid shape for the current tensor");

		_layer_dim = new_shape.x;
		_row_dim = new_shape.y;
		_col_dim = new_shape.z;

		return *this;
	}

	/// <summary>
	/// Returns size of convolution result in certain dimension
	/// </summary>
	/// <param name="in_size">Input size (in the chosen dimension)</param>
	/// <param name="kernel_size">Kernel size (in the chosen dimension)</param>
	/// <param name="padding">Padding size (in the chosen dimension)</param>
	/// <param name="stride">Stride size (in the chosen dimension)</param>
	long long calc_out_size_for_convolution(const std::size_t in_size, const std::size_t kernel_size,
		const long long padding, const long long stride)
	{
		const auto temp = static_cast<long long>(in_size) + 2 * padding - static_cast<long long>(kernel_size) + 1ll;
		if (temp < 0)
			return 0;

		return  temp / stride;
	}

	inline Index3d Tensor::calc_conv_res_size(const Index3d& tensor_size, const Index3d& kernel_size, const Index3d& paddings, const Index3d& strides)
	{
		return { calc_out_size_for_convolution(tensor_size.x, kernel_size.x, paddings.x, strides.x),
				 calc_out_size_for_convolution(tensor_size.y, kernel_size.y, paddings.y, strides.y),
				 calc_out_size_for_convolution(tensor_size.z, kernel_size.z, paddings.z, strides.z) };
	}

	/// <summary>
	/// Linear rectifier function of integer argument
	/// </summary>
	inline long long relu(const long long x)
	{
		return x > 0ll ? x : 0ll;
	}

	/// <summary>
	/// Linear rectifier function, 3d version
	/// </summary>
	inline Index3d relu(const Index3d& v)
	{
		return { relu(v.x), relu(v.y), relu(v.z) };
	}

#define KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets, action)				\
			for (auto k_x = kernel_start_offsets.x; k_x < kernel_stop_offsets.x; k_x++)				\
			{																						\
				const auto t_x = tensor_offsets.x + k_x;											\
				for (auto k_y = kernel_start_offsets.y; k_y < kernel_stop_offsets.y; k_y++)			\
				{																					\
					const auto t_y = tensor_offsets.y + k_y;										\
					for (auto k_z = kernel_start_offsets.z; k_z < kernel_stop_offsets.z; k_z++)		\
					{																				\
						const auto t_z = tensor_offsets.z + k_z;									\
						action;																		\
					}																				\
				}																					\
			}


	/// <summary>
	/// Calculates offsets needed to calculate convolution result item with the given offsets (3d index)
	/// </summary>
	/// <param name="conv_res_offsets">Offset (3d index) of the convolution result item</param>
	/// <param name="tensor_size">Size of the tensor the convolution is applied to</param>
	/// <param name="kernel_size">Size of the convolution kernel</param>
	/// <param name="paddings">Zero paddings to be applied to the tensor</param>
	/// <param name="strides">Strides to be used when shifting convolution kernel over the tensor</param>
	/// <returns>Tuple consisting of "tensor_offsets, kernel_start_offsets, kernel_stop_offsets" in the exact same order</returns>
	inline std::tuple<Index3d, Index3d, Index3d> calc_kernel_loop_offsets(const Index3d& conv_res_offsets, const Index3d& tensor_size,
																		  const Index3d& kernel_size, const Index3d& paddings, const Index3d& strides)
	{
		const auto tensor_offsets = conv_res_offsets.hadamard_prod(strides) - paddings;
		const auto kernel_start_offsets = relu(-tensor_offsets);
		const auto kernel_stop_offsets = kernel_size - relu(tensor_offsets + kernel_size - tensor_size);

		return std::make_tuple(tensor_offsets, kernel_start_offsets, kernel_stop_offsets);
	}

	Index3d Tensor::convolve(RealMemHandle result_handle, const Tensor& kernel, const Index3d& paddings, const Index3d& strides) const
	{
		const auto kernel_size = kernel.size_3d();
		const auto tensor_size = size_3d();
		const auto result_size = calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (result_handle.size() != result_size.x * result_size.y * result_size.z)
			throw std::exception("Unexpected amount of memory to store the result");

		for (std::size_t res_data_id = 0; res_data_id < result_handle.size(); res_data_id++)
		{
			const auto result_offsets = data_id_to_index_3d_internal(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			Real part_res = Real(0);

			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				part_res += _data[coords_to_data_id(t_x, t_y, t_z)] * kernel(k_x, k_y, k_z);)

			result_handle[res_data_id] = part_res;
		}

		return result_size;
	}

	Tensor Tensor::convolve(const Tensor& kernel, const Index3d& paddings, const Index3d& strides) const
	{
		const auto result_dim = calc_conv_res_size(size_3d(), kernel.size_3d(), paddings, strides);
		auto result = Tensor(result_dim, false);

		convolve(result.get_handle(), kernel, paddings, strides);

		return result;
	}

	Index3d Tensor::pool(RealMemHandle result_handle, const PoolOperator& pool_operator, const Index3d& paddings,
		const Index3d& strides) const
	{
		const auto kernel_size = pool_operator.size_3d();
		const auto tensor_size = size_3d();
		const auto result_size = calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (result_handle.size() != result_size.x * result_size.y * result_size.z)
			throw std::exception("Unexpected amount of memory to store the result");

		for (std::size_t res_data_id = 0; res_data_id < result_handle.size(); res_data_id++)
		{
			const auto result_offsets = data_id_to_index_3d_internal(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			auto operator_clone = pool_operator.clone();

			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				operator_clone->add({ k_x, k_y, k_z }, _data[coords_to_data_id(t_x, t_y, t_z)]);)

			result_handle[res_data_id] = operator_clone->pool();
		}

		return result_size;
	}

	Tensor Tensor::pool(const PoolOperator& pool_operator, const Index3d& paddings, const Index3d& strides) const
	{
		const auto result_dim = calc_conv_res_size(size_3d(), pool_operator.size_3d(), paddings, strides);
		auto result = Tensor(result_dim, false);

		pool(result.get_handle(), pool_operator, paddings, strides);

		return result;
	}

	std::tuple<Tensor, Tensor> Tensor::convolution_gradient(const RealMemHandleConst& conv_res_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		const auto tensor_size = size_3d();
		const auto kernel_size = kernel.size_3d();
		const auto conv_result_size = calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (conv_res_grad.size() != conv_result_size.x * conv_result_size.y * conv_result_size.z)
			throw std::exception("Unexpected size of the convolution result gradient");

		auto kern_grad = Tensor(kernel_size, true);
		auto in_grad = Tensor(tensor_size, true);

		for (std::size_t res_data_id = 0; res_data_id < conv_res_grad.size(); res_data_id++)
		{
			const auto result_offsets = data_id_to_index_3d_internal(res_data_id, conv_result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			const auto factor = conv_res_grad[res_data_id];
			if (factor == Real(0))
				continue;

			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				kern_grad(k_x, k_y, k_z) += _data[coords_to_data_id(t_x, t_y, t_z)] * factor;
			in_grad(t_x, t_y, t_z) += kernel(k_x, k_y, k_z) * factor;)
		}

		return { kern_grad, in_grad };
	}

	std::tuple<Tensor, Tensor> Tensor::convolution_gradient(const Tensor& conv_res_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		if (calc_conv_res_size(size_3d(), kernel.size_3d(), paddings, strides) != conv_res_grad.size_3d())
			throw std::exception("Inconsistent input data.");

		return convolution_gradient(conv_res_grad.get_handle(), kernel, paddings, strides);
	}

	Tensor Tensor::pool_input_gradient(const RealMemHandleConst& pool_res_grad, const PoolOperator& pool_operator, const Index3d& paddings,
		const Index3d& strides) const
	{
		const auto tensor_size = size_3d();
		const auto kernel_size = pool_operator.size_3d();
		const auto conv_result_size = calc_conv_res_size(tensor_size, kernel_size, paddings, strides);;

		if (conv_result_size.x * conv_result_size.y * conv_result_size.z != pool_res_grad.size())
			throw std::exception("Unexpected size of the pool result gradient");

		auto in_grad = Tensor(tensor_size, true);

		for (std::size_t res_data_id = 0; res_data_id < pool_res_grad.size(); res_data_id++)
		{
			const auto result_offsets = data_id_to_index_3d_internal(res_data_id, conv_result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			const auto factor = pool_res_grad[res_data_id];
			if (factor == Real(0))
				continue;

			auto pool_operator_clone = pool_operator.clone();

			//Make the agent familiar with the items in the current window
			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				pool_operator_clone->add({ k_x, k_y, k_z }, _data[coords_to_data_id(t_x, t_y, t_z)]);)

				//Add derivatives calculated by the agent
				KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
					in_grad(t_x, t_y, t_z) += pool_operator_clone->pool_deriv({ k_x, k_y, k_z }) * factor;)
		}

		return in_grad;
	}

	Tensor Tensor::pool_input_gradient(const Tensor& pool_res_grad, const PoolOperator& pool_operator, const Index3d& paddings,
		const Index3d& strides) const
	{
		if (calc_conv_res_size(size_3d(), pool_operator.size_3d(), paddings, strides) != pool_res_grad.size_3d())
			throw std::exception("Inconsistent input data.");

		return pool_input_gradient(pool_res_grad.get_handle(), pool_operator, paddings, strides);
	}

	void Tensor::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		std::vector<Real> proxy;
		msgpack::type::make_define_array(_layer_dim, _row_dim, _col_dim, proxy).msgpack_unpack(msgpack_o);
		_data = reinterpret_cast<Real*>(std::malloc(size() * sizeof(Real)));
		std::copy(proxy.begin(), proxy.end(), begin());
	}

	RealMemHandleConst Tensor::get_handle() const
	{
		return RealMemHandleConst(_data, size());
	}

	RealMemHandle Tensor::get_handle()
	{
		return RealMemHandle(_data, size());
	}

	RealMemHandleConst Tensor::get_layer_handle(const std::size_t& layer_id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, 0, 0))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return RealMemHandleConst(_data + coords_to_data_id(layer_id, 0, 0), _row_dim*_col_dim);
	}

	RealMemHandle Tensor::get_layer_handle(const std::size_t& layer_id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, 0, 0))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return RealMemHandle(_data + coords_to_data_id(layer_id, 0, 0), _row_dim * _col_dim);
	}
}