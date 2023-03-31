//Copyright (c) 2023 Denys Dragunov, dragunovdenis@gmail.com
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
#include "../VectorNd.h"
#include <functional>

namespace DeepLearning
{
	/// <summary>
	/// Component to do Nelder-Mead optimization (aka "amoeba optimization")
	/// </summary>
	template <int N>
	class NelderMeadOptimizer
	{
	public:
		using CostFunc = std::function<Real(const VectorNdReal<N>&)>;
		using DiagnosticsFunc = std::function<void(const std::array<VectorNdReal<N>, N + 1>& simplex,
			const int min_vertex_id, const Real& simplex_size)>;
	private:
		static_assert(N > 0, "Invalid dimension");

		/// <summary>
		/// Vertices of the current simplex
		/// </summary>
		std::array<VectorNdReal<N>, N + 1> _simplex{};

		/// <summary>
		/// Values of the cost function on the vertices of the "current" simplex
		/// </summary>
		std::array<Real, N + 1> _func_values{};

		/// <summary>
		/// Lower bound restriction of the coordinates of simplex vertices
		/// </summary>
		VectorNdReal<N> _constraints_lower{};

		/// <summary>
		/// Upper bound restriction of the coordinates of simplex vertices
		/// </summary>
		VectorNdReal<N> _constraints_upper{};

		/// <summary>
		/// Reflection coefficient
		/// </summary>
		Real _rho{};

		/// <summary>
		/// Expansion coefficient
		/// </summary>
		Real _chi{};

		/// <summary>
		/// Contraction coefficient
		/// </summary>
		Real _gamma{};

		/// <summary>
		/// Shrinkage coefficient
		/// </summary>
		Real _sigma{};

		/// <summary>
		/// Optimization process terminates when simplex becomes smaller than this value
		/// </summary>
		Real _min_simplex_size = std::numeric_limits<Real>::epsilon();

		/// <summary>
		/// Index of the vertex with the minimal value of the cost function (among all the simplex vertices)
		/// </summary>
		int _min_vertex_id = -1;

		/// <summary>
		/// Index of the vertex with the maximal value of the cost function (among all the simplex vertices)
		/// </summary>
		int _max_vertex_id = -1;

		/// <summary>
		/// Returns maximal value of the cost function on the vertices of the simplex
		/// </summary>
		[[nodiscard]] Real get_max_value() const
		{
			return _func_values[_max_vertex_id];
		}

		/// <summary>
		/// Index of the vertex with the second maximal value of the cost function (among all the vertices)
		/// </summary>
		int _second_max_vertex_id = -1;

		/// <summary>
		/// Returns the second biggest value of the cost function on the vertices of the simplex
		/// </summary>
		[[nodiscard]] Real get_second_max_value() const
		{
			return _func_values[_second_max_vertex_id];
		}

		/// <summary>
		/// Updates position of the vertex associated with the highest cost function value as well as the value itself
		///	After this procedure basic vertex indices must be updated (`update_basic_vertex_indices()`)
		/// </summary>
		void update_max_value_vertex(const VectorNdReal<N>& new_vertex_pos, const Real& new_value)
		{
			//this specific way of updating simplex with a new vertex
			//substituting the one previously "bearing" the highest value of the cost function
			//is coherent with the subroutine that calculates indices of the
			//"max" and "second max" vertices (`update_basic_vertex_indices()`) in order to ensure the "tie-braking rule"
			for (auto i = _max_vertex_id; i > 0; --i)
			{
				_func_values[i] = _func_values[i - 1];
				_simplex[i] = _simplex[i - 1];
			}

			_func_values[0] = new_value;
			_simplex[0] = new_vertex_pos;
		}

		/// <summary>
		/// Updates indices of the min, max, and second max (in terms of the cost function value) vertices of the simplex
		/// </summary>
		void update_basic_vertex_indices()
		{
			auto min_value = std::numeric_limits<Real>::max();
			auto max_value = -std::numeric_limits<Real>::max();
			auto second_max_value = max_value;

			_min_vertex_id = -1;
			_max_vertex_id = -1;
			_second_max_vertex_id = -1;

			for (auto val_id = 0; val_id < N + 1; ++val_id)
			{
				const auto val = _func_values[val_id];

				if (min_value >= val)
				{
					min_value = val;
					_min_vertex_id = val_id;
				}

				if (max_value < val)
				{
					_second_max_vertex_id = _max_vertex_id;
					second_max_value = max_value;

					max_value = val;
					_max_vertex_id = val_id;
					
				} else if (second_max_value < val)
				{
					second_max_value = val;
					_second_max_vertex_id = val_id;
				}
			}

			if (_min_vertex_id == -1 || _max_vertex_id == -1 || _second_max_vertex_id == -1)
				throw std::exception("Failed to update basic vertex indices");
		}

		/// <summary>
		/// Updates values of the given function on the vertices of the "current" simplex
		/// </summary>
		void update_function_values_on_simplex(const CostFunc& func)
		{
			for (auto vertex_id = 0; vertex_id < N + 1; ++vertex_id)
				_func_values[vertex_id] = func(_simplex[vertex_id]);
		}

		/// <summary>
		/// Adjusts the given vertex so that it fulfills constraints;
		/// </summary>
		[[nodiscard]] VectorNdReal<N> fulfill_constraints(const VectorNdReal<N>& vertex) const
		{
			auto result = vertex;
			for (auto i = 0; i < N; ++i)
			{
				if (result[i] < _constraints_lower[i])
					result[i] = _constraints_lower[i];
				else if (result[i] > _constraints_upper[i])
					result[i] = _constraints_upper[i];
			}

			return result;
		}

		/// <summary>
		/// Adjusts vertices of the given simplex so that they fulfill constraints;
		/// </summary>
		void fulfill_constraints(std::array<VectorNdReal<N>, N + 1>& simplex) const
		{
			for (auto i = 0; i < N + 1; ++i)
				simplex[i] = fulfill_constraints(simplex[i]);
		}

		/// <summary>
		/// Returns the length of the longest edge of the current simplex
		/// </summary>
		[[nodiscard]] Real calc_simplex_size() const
		{
			auto result = static_cast<Real>(0);

			for (auto i = 0; i < N + 1; ++i)
				for (auto j = i + 1; j < N + 1; ++j)
					result = std::max(result, (_simplex[i] - _simplex[j]).length());

			return result;
		}

		/// <summary>
		/// Calculates centroid of all the vertices of the simplex
		/// except for the one that corresponds to the maximal value of the cost function
		/// </summary>
		[[nodiscard]] VectorNdReal<N> calc_centroid() const
		{
			VectorNdReal<N> result{};

			for (auto i = 0; i < N + 1; ++i)
			{
				if (i == _max_vertex_id)
					continue;

				result += _simplex[i];
			}

			result *= static_cast<Real>(1) / N;

			return result;
		}

		/// <summary>
		/// A "move" of amoeba
		/// </summary>
		[[nodiscard]] VectorNdReal<N> move(const VectorNdReal<N>& centroid, const Real& factor) const
		{
			return fulfill_constraints((static_cast<Real>(1) + factor) * centroid - factor * _simplex[_max_vertex_id]);
		}

		/// <summary>
		/// Performs shrinkage 
		/// </summary>
		void shrink(const CostFunc& func)
		{
			const auto& min_vertex = _simplex[_min_vertex_id];
			for (auto i = 0; i < N + 1; ++i)
			{
				if (i == _min_vertex_id)
					continue;

				_simplex[i] = fulfill_constraints(min_vertex + _sigma * (_simplex[i] - min_vertex));
				_func_values[i] = func(_simplex[i]);
			}
		}

	public:
		/// <summary>
		/// Returns minimal value of the cost function on the vertices of the simplex
		/// </summary>
		[[nodiscard]] Real get_min_value() const
		{
			return _func_values[_min_vertex_id];
		}

		/// <summary>
		/// Returns vertex of the simplex corresponding to the minimal value of the cost function
		/// </summary>
		[[nodiscard]] VectorNdReal<N> get_min_vertex() const
		{
			return _simplex[_min_vertex_id];
		}

		/// <summary>
		/// Returns regular simplex in N-dimensional space with edges of the given length,
		/// containing the given initial point as its first vertex
		/// </summary>
		static std::array<VectorNdReal<N>, N + 1> create_regular_simplex(const VectorNdReal<N>& init_pt, const Real& edge_length)
		{
			std::array<VectorNdReal<N>, N + 1> result;
			result[0] = init_pt;

			const auto factor = static_cast<Real>(1) / (N * std::sqrt(static_cast<Real>(2)));
			const auto radical = std::sqrt(static_cast<Real>(N + 1));

			const auto p = factor * (N - 1 + radical);
			const auto q = factor * (radical - 1);

			for (auto vertex_id = 1; vertex_id < N + 1; ++vertex_id)
			{
				auto& vertex = result[vertex_id] = result[0];

				for (auto elem_id = 0; elem_id < N; ++elem_id)
					vertex[elem_id] = vertex[elem_id] + ((elem_id == vertex_id - 1) ? edge_length * p : edge_length * q);
			}

			return result;
		}

		/// <summary>
		/// Returns an axes aligned simplex in N-dimensional space with axes parallel edges of the given length,
		/// containing the given initial point as its first vertex
		/// </summary>
		static std::array<VectorNdReal<N>, N + 1> create_axes_aligned_simplex(const VectorNdReal<N>& init_pt, const Real& edge_length)
		{
			std::array<VectorNdReal<N>, N + 1> result;
			result[0] = init_pt;

			for (auto vertex_id = 1; vertex_id < N + 1; ++vertex_id)
			{
				auto& vertex = result[vertex_id] = result[0];
				vertex[vertex_id - 1] += edge_length;
			}

			return result;
		}

		/// <summary>
		/// Constructor
		/// </summary>
		NelderMeadOptimizer(const Real rho = static_cast<Real>(1),
			const Real chi = static_cast<Real>(2),
			const Real gamma = static_cast<Real>(0.5),
			const Real sigma = static_cast<Real>(0.5),
			const Real min_simplex_size = static_cast<Real>(1e-6))
		{
			set_rho_and_chi(rho, chi);
			set_gamma(gamma);
			set_sigma(sigma);
			set_min_simplex_size(min_simplex_size);
			//Essentially no constraints by default
			_constraints_lower = VectorNdReal<N>(-std::numeric_limits<Real>::max());
			_constraints_upper = VectorNdReal<N>(std::numeric_limits<Real>::max());
		}

		/// <summary>
		/// Setter for the corresponding coefficients
		/// </summary>
		void set_rho_and_chi(const Real& rho, const Real& chi)
		{
			if (rho <= 0 || chi <= rho)
				throw std::exception("Parameter Rho must be positive and less than Chi");

			_rho = rho;
			_chi = chi;
		}

		/// <summary>
		/// Getter for the corresponding coefficient
		/// </summary>
		[[nodiscard]] Real get_rho() const
		{
			return _rho;
		}

		/// <summary>
		/// Getter for the corresponding coefficient
		/// </summary>
		[[nodiscard]] Real get_chi() const
		{
			return _chi;
		}

		/// <summary>
		/// Setter for the corresponding coefficient
		/// </summary>
		void set_gamma(const Real& gamma)
		{
			if (gamma <= static_cast<Real>(0) || gamma >= static_cast<Real>(1))
				throw std::exception("Parameter Gamma must be positive and less than 1");

			_gamma = gamma;
		}

		/// <summary>
		/// Getter for the corresponding coefficient
		/// </summary>
		[[nodiscard]] Real get_gamma() const
		{
			return _gamma;
		}

		/// <summary>
		/// Setter for the corresponding coefficient
		/// </summary>
		void set_sigma(const Real& sigma)
		{
			if (sigma <= static_cast<Real>(0) || sigma >= static_cast<Real>(1))
				throw std::exception("Parameter Sigma must be positive and less than 1");

			_sigma = sigma;
		}

		/// <summary>
		/// Getter for the corresponding coefficient
		/// </summary>
		[[nodiscard]] Real get_sigma() const
		{
			return _sigma;
		}

		/// <summary>
		/// Setter for the corresponding field
		/// </summary>
		void set_min_simplex_size(const Real& min_simplex_size)
		{
			_min_simplex_size = min_simplex_size;
		}

		/// <summary>
		/// Getter for the corresponding field
		/// </summary>
		[[nodiscard]] Real get_min_simplex_size() const
		{
			return _min_simplex_size;
		}

		/// <summary>
		/// Setter for the lower constraints vector
		/// </summary>
		void set_constraints_lower(const VectorNdReal<N>& constraints)
		{
			_constraints_lower = constraints;
		}

		/// <summary>
		/// Getter for the lower constraints vector
		/// </summary>
		[[nodiscard]] VectorNdReal<N> get_constraints_lower() const
		{
			return _constraints_lower;
		}

		/// <summary>
		/// Setter for the upper constraints vector
		/// </summary>
		void set_constraints_upper(const VectorNdReal<N>& constraints)
		{
			_constraints_upper = constraints;
		}

		/// <summary>
		/// Getter for the upper constraints vector
		/// </summary>
		[[nodiscard]] VectorNdReal<N> get_constraints_upper() const
		{
			return _constraints_upper;
		}

		/// <summary>
		/// The optimization subroutine
		/// </summary>
		/// <param name="cost_func">Cost function</param>
		/// <param name="init_edge_length">Edge length of the initial simplex (has no effect if initialization is skipped)</param>
		/// <param name="init_pt">First point of the initial simplex (has no effect if initialization is skipped)</param>
		/// <param name="skip_initialization">Determines whether to do initialization or not. Should be set to false if, for example
		/// we want to continue optimization using a de-serialized object</param>
		/// <param name="regular_simplex">If "true" regular initial simplex will be used (has no effect if initialization is skipped)</param>
		/// <param name="diagnostics_func_ptr">Diagnostics call-back function</param>
		/// <param name="stop">Pointer to a stop token</param>
		void optimize(const CostFunc& cost_func,
		              Real init_edge_length = 1,
		              const VectorNdReal<N>& init_pt = VectorNdReal<N>(0),
		              const bool skip_initialization = false,
		              const bool regular_simplex = true,
					  const DiagnosticsFunc diagnostics_func_ptr = nullptr,
					  const std::stop_token* stop = nullptr)
		{
			if (!skip_initialization)
			{
				_simplex = regular_simplex ? create_regular_simplex(init_pt, init_edge_length) :
					create_axes_aligned_simplex(init_pt, init_edge_length);

				fulfill_constraints(_simplex);
				update_function_values_on_simplex(cost_func);
			}

			Real simplex_size;

			while ((simplex_size = calc_simplex_size()) > _min_simplex_size)
			{
				if (stop != nullptr && stop->stop_requested())
					break;

				update_basic_vertex_indices();

				if (diagnostics_func_ptr != nullptr)
					diagnostics_func_ptr(_simplex, _min_vertex_id, simplex_size);

				const auto centroid = calc_centroid();

				//Reflect
				const auto v_r = move(centroid, _rho);
				const auto f_r = cost_func(v_r);

				if (get_min_value() <= f_r && f_r < get_second_max_value())
				{
					update_max_value_vertex(v_r, f_r);
					continue;
				}

				if (f_r < get_min_value())
				{
					//Expand
					const auto v_e = move(centroid, _rho * _chi);
					const auto f_e = cost_func(v_e);

					if (f_e < f_r)
						update_max_value_vertex(v_e, f_e);
					else
						update_max_value_vertex(v_r, f_r);

					continue;
				}

				if (f_r >= get_second_max_value())
				{
					//Contract
					if (f_r < get_max_value())
					{
						//Outside contraction
						const auto v_c = move(centroid, _rho * _gamma);
						const auto f_c = cost_func(v_c);

						if (f_c <= f_r)
						{
							update_max_value_vertex(v_c, f_c);
							continue;
						}

						shrink(cost_func);
						continue;
					}

					//Inside contraction
					const auto v_cc = move(centroid, -_gamma);
					const auto f_cc = cost_func(v_cc);

					if (f_cc < get_max_value())
					{
						update_max_value_vertex(v_cc, f_cc);
						continue;
					}

					shrink(cost_func);
				}
			}

			//Update the indices so that min value point can be
			//accessed by the caller through the corresponding interface method
			update_basic_vertex_indices();
		}
	};
}
