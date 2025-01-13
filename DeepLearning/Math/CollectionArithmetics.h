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
#include <vector>

namespace DeepLearning
{
	/// <summary>
	/// Compound collection addition operator
	/// </summary>
	/// <param name="op1">Operand to add to</param>
	/// <param name="op2">Operand to be added to</param>
	/// <returns>Reference to the modified operand</returns>
	template <class T>
	std::vector<T>& operator +=(std::vector<T>& op1, const std::vector<T>& op2)
	{
		if (op1.size() != op2.size())
			throw std::exception("Operands have incompatible sizes");

		for (auto item_id = 0ull; item_id < op1.size(); item_id++)
			op1[item_id] += op2[item_id];

		return op1;
	}

	/// <summary>
	/// Collection addition operator.
	/// </summary>
	template <class T>
	std::vector<T> operator +(std::vector<T> op1, const std::vector<T>& op2)
	{
		return op1 += op2;
	}

	/// <summary>
	/// Compound collection subtraction operator
	/// </summary>
	/// <param name="op1">Operand to subtract from</param>
	/// <param name="op2">Operand to be subtracted</param>
	/// <returns>Reference to the modified operand</returns>
	template <class T>
	std::vector<T>& operator -=(std::vector<T>& op1, const std::vector<T>& op2)
	{
		if (op1.size() != op2.size())
			throw std::exception("Operands have incompatible sizes");

		for (auto item_id = 0ull; item_id < op1.size(); item_id++)
			op1[item_id] -= op2[item_id];

		return op1;
	}

	/// <summary>
	/// Compound collection scalar multiplication operator
	/// </summary>
	/// <param name="op1">Operand to add to</param>
	/// <param name="scalar">Scalar to multiply by</param>
	/// <returns>Reference to the modified operand</returns>
	template <class T>
	std::vector<T>& operator *=(std::vector<T>& op1, const Real& scalar)
	{
		for (auto item_id = 0ull; item_id < op1.size(); item_id++)
			op1[item_id] *= scalar;

		return op1;
	}
}
