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
	{
		resize(layer_dim, row_dim, col_dim);

		if (assign_zero)
			fill_zero();
	}

	Tensor::Tensor(const Index3d& size, const bool assign_zero)
		: Tensor(size.x, size.y, size.z, assign_zero)
	{}

	Tensor::Tensor(const Index3d& size, const Real range_begin, const Real range_end, std::mt19937* seeder)
		: Tensor(size.x, size.y, size.z, range_begin, range_end, seeder)
	{}

	Tensor::Tensor(const Tensor& tensor)
		: Tensor(tensor._layer_dim, tensor._row_dim, tensor._col_dim, false)
	{
		std::ranges::copy(tensor, begin());
	}

	void Tensor::abandon_resources()
	{
		Base::abandon_resources();
		_layer_dim = 0;
		_row_dim = 0;
		_col_dim = 0;

	}

	Tensor::Tensor(Tensor&& tensor) noexcept
		: _layer_dim(tensor._layer_dim),
		_row_dim(tensor._row_dim), _col_dim(tensor._col_dim)
	{
		take_over_resources(std::move(tensor));
	}

	Tensor::Tensor(const std::size_t layer_dim, const std::size_t row_dim,
		const std::size_t col_dim, const Real range_begin, const Real range_end, std::mt19937* seeder)
		: Tensor(layer_dim, row_dim, col_dim, false)
	{
		uniform_random_fill(range_begin, range_end, seeder);
	}

	Tensor& Tensor::operator =(const Tensor& tensor)
	{
		if (this != &tensor)
		{
			resize(tensor.size_3d());
			std::ranges::copy(tensor, begin());
		}

		return *this;
	}

	Tensor::Tensor(Vector&& vector) noexcept : _layer_dim(1ull),
		_row_dim(1ull), _col_dim(vector.dim())
	{
		take_over_resources(std::move(vector));
	}

	Tensor::Tensor(Matrix&& matrix) noexcept : _layer_dim(1ull),
		_row_dim(matrix.row_dim()), _col_dim(matrix.col_dim())
	{
		take_over_resources(std::move(matrix));
	}

	Tensor& Tensor::operator =(Vector&& vector) noexcept
	{
		_layer_dim = 1ull;
		_row_dim = 1ull;
		_col_dim = vector.dim();
		take_over_resources(std::move(vector));

		return *this;
	}

	Tensor& Tensor::operator =(Matrix&& matrix) noexcept
	{
		_layer_dim = 1ull;
		_row_dim = matrix.row_dim();
		_col_dim = matrix.col_dim();
		take_over_resources(std::move(matrix));

		return *this;
	}

	Tensor& Tensor::operator =(Tensor&& tensor) noexcept
	{
		if (this != &tensor)
		{
			_layer_dim = tensor.layer_dim();
			_row_dim = tensor.row_dim();
			_col_dim = tensor.col_dim();
			take_over_resources(std::move(tensor));
		}

		return *this;
	}

	void Tensor::resize(const std::size_t& new_layer_dim, const std::size_t& new_row_dim, const std::size_t& new_col_dim)
	{
		const auto new_size = new_layer_dim * new_row_dim * new_col_dim;
		allocate(new_size);
		_layer_dim = new_layer_dim;
		_row_dim = new_row_dim;
		_col_dim = new_col_dim;
	}

	void Tensor::resize(const Index3d& size_3d)
	{
		resize(size_3d.x, size_3d.y, size_3d.z);
	}

	Tensor& Tensor::get_resized(const Index3d& size_3d)
	{
		resize(size_3d);
		return *this;
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

		return begin()[coords_to_data_id(layer_id, row_id, col_id)];
	}

	const Real& Tensor::operator ()(const std::size_t layer_id, const std::size_t row_id, const std::size_t col_id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, row_id, col_id))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return begin()[coords_to_data_id(layer_id, row_id, col_id)];
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
		const auto data = begin();
		const auto tensor_data = tensor.begin();
		return _layer_dim == tensor._layer_dim &&
			_row_dim == tensor._row_dim &&
			_col_dim == tensor._col_dim &&
			std::all_of(IndexIterator(0), IndexIterator(static_cast<int>(size())),
				[&](const auto& id) { return data[id] == tensor_data[id]; });
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

		const auto data = begin();

		for (std::size_t res_data_id = 0; res_data_id < result_handle.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			double part_res = 0;

			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				part_res += data[coords_to_data_id(t_x, t_y, t_z)] * kernel(k_x, k_y, k_z);)

			result_handle[res_data_id] = static_cast<Real>(part_res);
		}

		return result_size;
	}

	void Tensor::convolve(Tensor& result, const std::vector<Tensor>& kernels, const Index3d& paddings, const Index3d& strides) const
	{
		if (result.layer_dim() != kernels.size())
			throw std::exception("Inconsistent input");

		for (auto kernel_id = 0ull; kernel_id < kernels.size(); kernel_id++)
			convolve(result.get_layer_handle(kernel_id), kernels[kernel_id], paddings, strides);
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

		if (result_handle.size() != result_size.coord_prod())
			throw std::exception("Unexpected amount of memory to store the result");

		const auto data = begin();

		for (std::size_t res_data_id = 0; res_data_id < result_handle.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			const auto operator_clone = pool_operator.clone();

			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				operator_clone->add({ k_x, k_y, k_z }, data[coords_to_data_id(t_x, t_y, t_z)]);)

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

	template <bool CALC_INPUT_GRAD>
	void Tensor::convolution_gradient(const RealMemHandleConst& conv_res_grad, Tensor& input_grad, Tensor& kernel_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides, const Real kernel_grad_scale) const
	{
		const auto tensor_size = size_3d();
		const auto kernel_size = kernel.size_3d();
		const auto conv_result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, kernel_size, paddings, strides);

		if (conv_res_grad.size() != conv_result_size.coord_prod())
			throw std::exception("Unexpected size of the convolution result gradient");

		if (CALC_INPUT_GRAD && input_grad.size_3d() != tensor_size)
			throw std::exception("Unexpected size of the input gradient container");

		if (kernel_grad_scale != static_cast<Real>(0))
			kernel_grad *= kernel_grad_scale;
		else
			kernel_grad.fill_zero();

		const auto data = begin();

		for (std::size_t res_data_id = 0; res_data_id < conv_res_grad.size(); res_data_id++)
		{
			const auto factor = conv_res_grad[res_data_id];
			if (factor == static_cast<Real>(0))
				continue;

			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, conv_result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				kernel_grad(k_x, k_y, k_z) += data[coords_to_data_id(t_x, t_y, t_z)] * factor;
			if (CALC_INPUT_GRAD)
				input_grad(t_x, t_y, t_z) += kernel(k_x, k_y, k_z) * factor;)
		}
	}

	template void Tensor::convolution_gradient<true>(const RealMemHandleConst& conv_res_grad, Tensor& input_grad,
		Tensor& kernel_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides, const Real kernel_grad_scale) const;
	template void Tensor::convolution_gradient<false>(const RealMemHandleConst& conv_res_grad, Tensor& input_grad,
		Tensor& kernel_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides, const Real kernel_grad_scale) const;

	template <bool CALC_INPUT_GRAD>
	Tensor Tensor::convolution_gradient(const RealMemHandleConst& conv_res_grad, Tensor& input_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		Tensor kernel_grad(kernel.size_3d(), false);
		convolution_gradient<CALC_INPUT_GRAD>(conv_res_grad, input_grad, kernel_grad, kernel, paddings, strides, static_cast<Real>(0));

		return kernel_grad;
	}

	template Tensor Tensor::convolution_gradient<true>(const RealMemHandleConst& conv_res_grad, Tensor& input_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides) const;
	template Tensor Tensor::convolution_gradient<false>(const RealMemHandleConst& conv_res_grad, Tensor& input_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides) const;

	std::tuple<Tensor, Tensor> Tensor::convolution_gradient(const Tensor& conv_res_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		if (ConvolutionUtils::calc_conv_res_size(size_3d(), kernel.size_3d(), paddings, strides) != conv_res_grad.size_3d())
			throw std::exception("Inconsistent input data.");

		Tensor input_grad(size_3d(), true);
		auto kernel_grad = convolution_gradient(conv_res_grad.get_handle(), input_grad, kernel, paddings, strides);

		return std::make_tuple(std::move(kernel_grad), std::move(input_grad));
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
		const auto data = begin();

		for (std::size_t res_data_id = 0; res_data_id < pool_res_grad.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, conv_result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, kernel_size, paddings, strides);

			const auto factor = pool_res_grad[res_data_id];
			if (factor == Real(0))
				continue;

			const auto pool_operator_clone = pool_operator.clone();

			//Make the agent familiar with the items in the current window
			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				pool_operator_clone->add({ k_x, k_y, k_z }, data[coords_to_data_id(t_x, t_y, t_z)]);)

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

	std::tuple<Tensor, std::vector<int>> Tensor::min_max_pool(const Index3d& window_size, const bool max) const
	{
		Tensor result;
		std::vector<int> index_map;
		min_max_pool<true>(window_size, max, result, index_map);
		return std::make_tuple(std::move(result), std::move(index_map));
	}

	void Tensor::min_max_pool(const Index3d& window_size, const bool max, Tensor& result) const
	{
		std::vector<int> index_map;
		min_max_pool<false>(window_size, max, result, index_map);
	}

	template <bool EVAL_MAP>
	void Tensor::min_max_pool(const Index3d& window_size, const bool max, Tensor& result, std::vector<int>& index_map) const
	{
		const auto paddings = Index3d{ 0 };
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, window_size, paddings, window_size);

		result.resize(result_size);

		if (EVAL_MAP)
			index_map.resize(result.size());

		const auto init_val = max ? -std::numeric_limits<Real>::max() : std::numeric_limits<Real>::max();

		const auto comparer = max ?
			static_cast<std::function<bool(const Real&, const Real&)>>([](const auto& a, const auto& b) { return a < b; }) :
			[](const auto& a, const auto& b) { return a > b; };

		const auto data = begin();
		const auto result_data = result.begin();

		for (std::size_t res_data_id = 0; res_data_id < result.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, window_size, paddings, window_size);

			auto poolled_val = init_val;
			auto poolled_id = -1;
			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				const auto tensor_data_id = static_cast<int>(coords_to_data_id(t_x, t_y, t_z));
				const auto & current_val = data[tensor_data_id];
				if (comparer(poolled_val, current_val))
				{
					poolled_val = current_val;
					poolled_id = tensor_data_id;
				});

			if (EVAL_MAP)
				index_map[res_data_id] = poolled_id;

			result_data[res_data_id] = poolled_val;
		}
	}

	template void Tensor::min_max_pool<true>(const Index3d& window_size, const bool max, Tensor& result, std::vector<int>& index_map) const;
	template void Tensor::min_max_pool<false>(const Index3d& window_size, const bool max, Tensor& result, std::vector<int>& index_map) const;

	void Tensor::min_max_pool_input_gradient(const Tensor& pool_res_gradient, const std::vector<int>& out_to_in_mapping, Tensor& result) const
	{
		if (pool_res_gradient.size() != out_to_in_mapping.size())
			throw std::exception("Inconsistent input");

		result.resize(size_3d());
		result.fill_zero();

		const auto result_data = result.begin();

		auto map_ptr = out_to_in_mapping.begin();
		for (auto grad_ptr = pool_res_gradient.begin(); grad_ptr != pool_res_gradient.end(); ++grad_ptr, ++map_ptr)
			result_data[*map_ptr] = *grad_ptr;
	}

	Tensor Tensor::min_max_pool_input_gradient(const Tensor& pool_res_gradient, const std::vector<int>& out_to_in_mapping) const
	{
	    Tensor result;
		min_max_pool_input_gradient(pool_res_gradient, out_to_in_mapping, result);
		return result;
	}

	Tensor Tensor::scale_pool(const Index3d& window_size, const Real& scale_factor) const
	{
		Tensor result;
		scale_pool(window_size, scale_factor, result);
		return result;
	}

	void Tensor::scale_pool(const Index3d& window_size, const Real& scale_factor, Tensor& result) const
	{
		const auto paddings = Index3d{ 0 };
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, window_size, paddings, window_size);

		result.resize(result_size);

		const auto data = begin();
		const auto result_data = result.begin();

		for (std::size_t res_data_id = 0; res_data_id < result.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, window_size, paddings, window_size);

			auto poolled_val = Real(0);
			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				poolled_val += data[coords_to_data_id(t_x, t_y, t_z)];);

			result_data[res_data_id] = poolled_val * scale_factor;
		}
	}

	void Tensor::scale_pool_input_gradient(const Tensor& pool_res_gradient, const Index3d& window_size, const Real& scale_factor, Tensor& result) const
	{
		const auto paddings = Index3d{ 0 };
		const auto tensor_size = size_3d();
		const auto result_size = ConvolutionUtils::calc_conv_res_size(tensor_size, window_size, paddings, window_size);

		if (result_size != pool_res_gradient.size_3d())
			throw std::exception("Unexpected size of the gradient tensor");

		result.resize(tensor_size);
		result.fill_zero();

		const auto pool_res_gradient_data = pool_res_gradient.begin();
		const auto result_data = result.begin();

		for (std::size_t res_data_id = 0; res_data_id < pool_res_gradient.size(); res_data_id++)
		{
			const auto result_offsets = ConvolutionUtils::data_id_to_index_3d(res_data_id, result_size);
			const auto [tensor_offsets, kernel_start_offsets, kernel_stop_offsets] =
				ConvolutionUtils::calc_kernel_loop_offsets(result_offsets, tensor_size, window_size, paddings, window_size);

			const auto value = pool_res_gradient_data[res_data_id] * scale_factor;
			KERNEL_LOOP(kernel_start_offsets, kernel_stop_offsets, tensor_offsets,
				//We assume that "pool windows" do not intersect (strides == wingow_size) and thus
				//use direct assignment ("=") instead of accumulation ("+=") in the line below
				result_data[coords_to_data_id(t_x, t_y, t_z)] = value;);
		}
	}

	Tensor Tensor::average_pool(const Index3d& window_size) const
	{
		return scale_pool(window_size, Real(1) / window_size.coord_prod());
	}

	void Tensor::average_pool(const Index3d& window_size, Tensor& result) const
	{
		scale_pool(window_size, Real(1) / window_size.coord_prod(), result);
	}

	void Tensor::average_pool_input_gradient(const Tensor& pool_res_gradient, const Index3d& window_size, Tensor& result) const
	{
		scale_pool_input_gradient(pool_res_gradient, window_size, Real(1) / window_size.coord_prod(), result);
	}

	Tensor Tensor::average_pool_input_gradient(const Tensor& pool_res_gradient, const Index3d& window_size) const
	{
		Tensor result;
		scale_pool_input_gradient(pool_res_gradient, window_size, Real(1) / window_size.coord_prod(), result);
		return result;
	}

	void Tensor::msgpack_unpack(msgpack::object const& msgpack_o)
	{
		std::vector<Real> proxy;
		std::size_t layer_dim, row_dim, col_dim;
		msgpack::type::make_define_array(layer_dim, row_dim, col_dim, proxy).msgpack_unpack(msgpack_o);
		resize(layer_dim, row_dim, col_dim);
		std::ranges::copy(proxy, begin());
	}

	RealMemHandleConst Tensor::get_layer_handle(const std::size_t& layer_id) const
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, 0, 0))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return RealMemHandleConst(begin() + coords_to_data_id(layer_id, 0, 0), _row_dim * _col_dim);
	}

	RealMemHandle Tensor::get_layer_handle(const std::size_t& layer_id)
	{
#ifdef CHECK_BOUNDS
		if (!check_bounds(layer_id, 0, 0))
			throw std::exception("Index out of bounds");
#endif // CHECK_BOUNDS

		return RealMemHandle(begin() + coords_to_data_id(layer_id, 0, 0), _row_dim * _col_dim);
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