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

	Tensor::Tensor(const Tensor& tensor)
		: Tensor(tensor._layer_dim, tensor._row_dim, tensor._col_dim, false)
	{
		std::copy(tensor.begin(), tensor.end(), begin());
	}

	Tensor::Tensor(Tensor&& tensor) noexcept
		: _data(tensor._data), _layer_dim(tensor._layer_dim),
		_row_dim(tensor._row_dim), _col_dim(tensor._col_dim)
	{
		tensor._data = nullptr;
		tensor._layer_dim = 0;
		tensor._row_dim = 0;
		tensor._col_dim = 0;
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

	Index3d Tensor::data_id_to_index_3d(const long long data_id) const
	{
		const auto temp = std::div(data_id, static_cast<long long>(_col_dim));
		const auto col_id = temp.rem;
		const auto temp1 = std::div(temp.quot, static_cast<long long>(_row_dim));
		const auto row_id = temp1.rem;
		const auto layer_id = temp1.quot;

		return { layer_id, row_id, col_id };
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

	Tensor Tensor::convolve(const Tensor& kernel, const Index3d& paddings, const Index3d& strides) const
	{
		const Index3d result_dim = { calc_out_size_for_convolution(_layer_dim, kernel._layer_dim, paddings.x, strides.x) ,
									 calc_out_size_for_convolution(_row_dim, kernel._row_dim, paddings.y, strides.y) ,
									 calc_out_size_for_convolution(_col_dim, kernel._col_dim, paddings.z, strides.z) };
		auto result = Tensor(result_dim.x, result_dim.y, result_dim.z, false);

		const auto tensor_size = size_3d();
		const auto kernel_size = kernel.size_3d();

		for (std::size_t res_data_id = 0; res_data_id < result.size(); res_data_id++)
		{
			const auto result_offsets = result.data_id_to_index_3d(res_data_id);
			const auto tensor_offsets = result_offsets.hadamard_prod(strides) - paddings;
			const auto kernel_start_offsets = relu(-tensor_offsets);
			const auto kernel_stop_offsets = kernel_size - relu(tensor_offsets + kernel_size - tensor_size);

			Real part_res = Real(0);

			for (auto k_x = kernel_start_offsets.x; k_x < kernel_stop_offsets.x; k_x++)
			{
				const auto t_x = tensor_offsets.x + k_x;
				for (auto k_y = kernel_start_offsets.y; k_y < kernel_stop_offsets.y; k_y++)
				{
					const auto t_y = tensor_offsets.y + k_y;
					for (auto k_z = kernel_start_offsets.z; k_z < kernel_stop_offsets.z; k_z++)
					{
						const auto t_z = tensor_offsets.z + k_z;
						part_res += _data[coords_to_data_id(t_x, t_y, t_z)] * kernel(k_x, k_y, k_z);
					}
				}
			}

			result._data[res_data_id] = part_res;
		}

		return result;
	}

	std::tuple<Tensor, Tensor> Tensor::convolution_kernel_gradient(const Tensor& conv_res_grad, const Tensor& kernel, const Index3d& paddings,
		const Index3d& strides) const
	{
		const auto kernel_size = kernel.size_3d();

		const Index3d result_size_check = { calc_out_size_for_convolution(_layer_dim, kernel_size.x, paddings.x, strides.x) ,
									        calc_out_size_for_convolution(_row_dim, kernel_size.y, paddings.y, strides.y) ,
									        calc_out_size_for_convolution(_col_dim, kernel_size.z, paddings.z, strides.z) };
		const auto conv_res_grad_size = conv_res_grad.size_3d();

		if (result_size_check != conv_res_grad_size)
			throw std::exception("Inconsistent input data.");

		auto kern_grad = Tensor(kernel_size.x, kernel_size.y, kernel_size.z, true);
		auto in_grad = Tensor(_layer_dim, _row_dim, _col_dim, true);

		const auto tensor_size = size_3d();

		for (std::size_t res_data_id = 0; res_data_id < conv_res_grad.size(); res_data_id++)
		{
			const auto result_offsets = conv_res_grad.data_id_to_index_3d(res_data_id);
			const auto tensor_offsets = result_offsets.hadamard_prod(strides) - paddings;
			const auto kernel_start_offsets = relu(-tensor_offsets);
			const auto kernel_stop_offsets = kernel_size - relu(tensor_offsets + kernel_size - tensor_size);

			const auto factor = conv_res_grad._data[res_data_id];

			for (auto k_x = kernel_start_offsets.x; k_x < kernel_stop_offsets.x; k_x++)
			{
				const auto t_x = tensor_offsets.x + k_x;
				for (auto k_y = kernel_start_offsets.y; k_y < kernel_stop_offsets.y; k_y++)
				{
					const auto t_y = tensor_offsets.y + k_y;
					for (auto k_z = kernel_start_offsets.z; k_z < kernel_stop_offsets.z; k_z++)
					{
						const auto t_z = tensor_offsets.z + k_z;
						kern_grad(k_x, k_y, k_z) += _data[coords_to_data_id(t_x, t_y, t_z)] * factor;
						in_grad(t_x, t_y, t_z) += kernel(k_x, k_y, k_z) * factor;
					}
				}
			}
		}

		return { kern_grad, in_grad };
	}
}