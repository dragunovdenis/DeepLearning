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
#include "../Diagnostics/Logging.h"
#include "ConvolutionUtils.h"

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

	Tensor::Tensor(const Index3d& size, const Real range_begin, const Real range_end)
		: Tensor(size.x, size.y, size.z, range_begin, range_end)
	{}

	Tensor::Tensor(const Tensor& tensor)
		: Tensor(tensor._layer_dim, tensor._row_dim, tensor._col_dim, false)
	{
		std::copy(tensor.begin(), tensor.end(), begin());
	}

	void Tensor::abandon_resources()
	{
		_data = nullptr;
		free();
	}

	Tensor::Tensor(Tensor&& tensor) noexcept
		: _layer_dim(tensor._layer_dim),
		_row_dim(tensor._row_dim), _col_dim(tensor._col_dim)
	{
		_data = tensor._data;
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
		_row_dim(1ull), _col_dim(vector.dim())
	{
		_data = vector.begin();
		vector.abandon_resources();
	}

	/// <summary>
	/// Move constructor
	/// </summary>
	Tensor::Tensor(Matrix&& matrix) noexcept : _layer_dim(1ull),
		_row_dim(matrix.row_dim()), _col_dim(matrix.col_dim())
	{
		_data = matrix.begin();
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

	Index3d Tensor::data_id_to_index_3d(const long long data_id) const
	{
		return ConvolutionUtils::data_id_to_index_3d(data_id, size_3d());
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

	Index3d Tensor::convolve(RealMemHandle result_handle, const Tensor& kernel, const Index3d& paddings, const Index3d& strides) const
	{
		const auto kernel_size = kernel.size_3d();
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (result_handle.size() != result_size.x * result_size.y * result_size.z)
			throw std::exception("Unexpected amount of memory to store the result");

		for (std::size_t res_data_id = 0; res_data_id < result_handle.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			Real part_res = Real(0);

			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				part_res += _data[coords_to_data_id(t_x, t_y, t_z)] * kernel(k_x, k_y, k_z);)

			result_handle[res_data_id] = part_res;
		}

		return result_size;
	}

	Tensor Tensor::convolve(const Tensor& kernel, const Index3d& paddings, const Index3d& strides) const
	{
		const auto result_dim = ConvolutionUtils::calc_conv_res_size(size_3d(), kernel.size_3d(), paddings, strides);
		auto result = Tensor(result_dim, false);

		convolve(result.get_handle(), kernel, paddings, strides);

		return result;
	}

	Index3d Tensor::pool(RealMemHandle result_handle, const PoolOperator& pool_operator, const Index3d& paddings,
		const Index3d& strides) const
	{
		const auto kernel_size = pool_operator.size_3d();
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (result_handle.size() != result_size.x * result_size.y * result_size.z)
			throw std::exception("Unexpected amount of memory to store the result");

		for (std::size_t res_data_id = 0; res_data_id < result_handle.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			auto operator_clone = pool_operator.clone();

			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				operator_clone->add({ k_x, k_y, k_z }, _data[coords_to_data_id(t_x, t_y, t_z)]);)

			result_handle[res_data_id] = operator_clone->pool();
		}

		return result_size;
	}

	Tensor Tensor::pool(const PoolOperator& pool_operator, const Index3d& paddings, const Index3d& strides) const
	{
		const auto result_dim = ConvolutionUtils::calc_conv_res_size(size_3d(), pool_operator.size_3d(), paddings, strides);
		auto result = Tensor(result_dim, false);

		pool(result.get_handle(), pool_operator, paddings, strides);

		return result;
	}

	std::tuple<Tensor, Tensor> Tensor::convolution_gradient(const RealMemHandleConst& conv_res_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		const auto tensor_size = size_3d();
		const auto kernel_size = kernel.size_3d();
		const auto conv_result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (conv_res_grad.size() != conv_result_size.x * conv_result_size.y * conv_result_size.z)
			throw std::exception("Unexpected size of the convolution result gradient");

		auto kern_grad = Tensor(kernel_size, true);
		auto in_grad = Tensor(tensor_size, true);

		for (std::size_t res_data_id = 0; res_data_id < conv_res_grad.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, conv_result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

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
		if (ConvolutionUtils::calc_conv_res_size(size_3d(), kernel.size_3d(), paddings, strides) != conv_res_grad.size_3d())
			throw std::exception("Inconsistent input data.");

		return convolution_gradient(conv_res_grad.get_handle(), kernel, paddings, strides);
	}

	Tensor Tensor::pool_input_gradient(const RealMemHandleConst& pool_res_grad, const PoolOperator& pool_operator, const Index3d& paddings,
		const Index3d& strides) const
	{
		const auto tensor_size = size_3d();
		const auto kernel_size = pool_operator.size_3d();
		const auto conv_result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);;

		if (conv_result_size.x * conv_result_size.y * conv_result_size.z != pool_res_grad.size())
			throw std::exception("Unexpected size of the pool result gradient");

		auto in_grad = Tensor(tensor_size, true);

		for (std::size_t res_data_id = 0; res_data_id < pool_res_grad.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, conv_result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

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
		if (ConvolutionUtils::calc_conv_res_size(size_3d(), pool_operator.size_3d(), paddings, strides) != pool_res_grad.size_3d())
			throw std::exception("Inconsistent input data.");

		return pool_input_gradient(pool_res_grad.get_handle(), pool_operator, paddings, strides);
	}

	std::tuple<Tensor, std::vector<std::size_t>> Tensor::min_max_pool_2d(const Index2d& window_size, const bool max) const
	{
		const auto paddings = Index3d{ 0 };
		const auto kernel_size = Index3d(1ll, window_size.x, window_size.y);
		const auto strides = kernel_size;
			const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		auto result = Tensor(result_size, false);
		auto out_to_in_map = std::vector<std::size_t>(result.size());

		const auto init_val = max ? -std::numeric_limits<Real>::max() : std::numeric_limits<Real>::max();

		const auto comparer = max ? (std::function<bool(const Real&, const Real&)>)[](const auto& a, const auto& b) { return a < b; } : [](const auto& a, const auto& b) { return a > b; };

		for (std::size_t res_data_id = 0; res_data_id < result.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			auto pool_res = init_val;
			const auto t_x = tensor_offsets.x + kernel_start_offsets.x;
			for (auto k_y = kernel_start_offsets.y; k_y < kernel_stop_offsets.y; k_y++)
			{										
				const auto t_y = tensor_offsets.y + k_y;	
				for (auto k_z = kernel_start_offsets.z; k_z < kernel_stop_offsets.z; k_z++)
				{
					const auto t_z = tensor_offsets.z + k_z;

					const auto tensor_data_id = coords_to_data_id(t_x, t_y, t_z);
					const auto& current_val = _data[tensor_data_id];
					if (comparer(pool_res, current_val))
					{
						pool_res = current_val;
						out_to_in_map[res_data_id] = tensor_data_id;
					}
				}
			}

			result._data[res_data_id] = pool_res;
		}

		return std::make_tuple(result, out_to_in_map);
	}

	Tensor Tensor::min_max_pool_2d_input_gradient(const Tensor& pool_res_gradient, const std::vector<std::size_t>& out_to_in_mapping) const
	{
		if (pool_res_gradient.size() != out_to_in_mapping.size())
			throw std::exception("Inconsistent input");

		auto result = Tensor(size_3d(), true/*zeros initialization*/);

		auto map_ptr = out_to_in_mapping.begin();
		for (auto grad_ptr = pool_res_gradient.begin(); grad_ptr != pool_res_gradient.end(); grad_ptr++, map_ptr++)
			result._data[*map_ptr] = *grad_ptr;

		return result;
	}

	void Tensor::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		std::vector<Real> proxy;
		msgpack::type::make_define_array(_layer_dim, _row_dim, _col_dim, proxy).msgpack_unpack(msgpack_o);
		_data = reinterpret_cast<Real*>(std::malloc(size() * sizeof(Real)));
		std::copy(proxy.begin(), proxy.end(), begin());
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

	std::vector<Tensor>& operator +=(std::vector<Tensor>& op1, const std::vector<Tensor>& op2)
	{
		if (op1.size() != op2.size())
			throw std::exception("Operands have incompatible sizes");

		for (auto item_id = 0ull; item_id < op1.size(); item_id++)
			op1[item_id] += op2[item_id];

		return op1;
	}

	std::vector<Tensor>& operator *=(std::vector<Tensor>& op1, const Real& scalar)
	{
		for (auto item_id = 0ull; item_id < op1.size(); item_id++)
			op1[item_id] *= scalar;

		return op1;
	}

	void Tensor::log_layer(const std::size_t& layer_id, const std::filesystem::path& filename) const
	{
		if (!check_bounds(layer_id, 0, 0))
			throw std::exception("Invalid layer ID");

		Logging::log_as_table(get_layer_handle(layer_id), row_dim(), col_dim(), filename);
	}

	void Tensor::log(const std::filesystem::path& directory, const std::filesystem::path& base_log_name) const
	{
		if (!std::filesystem::is_directory(directory))
			throw std::exception("Directory does not exist");

		for (auto layer_id = 0ull; layer_id < layer_dim(); layer_id++)
		{
			auto layer_log_file = directory / base_log_name;
			layer_log_file += "_" + std::to_string(layer_id) + ".txt";
			log_layer(layer_id, layer_log_file);
		}
	}
}