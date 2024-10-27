//Copyright (c) 2024 Denys Dragunov, dragunovdenis@gmail.com
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

namespace DeepLearning
{
	/// <summary>
	/// Data structure to hold results of cost function evaluation as well as the number of correct answers on some set of labeled data
	/// </summary>
	struct CostAndCorrectAnswers
	{
		/// <summary>
		/// Cost function value
		/// </summary>
		Real Cost{};

		/// <summary>
		/// Ratio of the correct answers to all the answers
		/// </summary>
		Real CorrectAnswers{};

		/// <summary>
		/// Compound addition operator
		/// </summary>
		CostAndCorrectAnswers& operator += (const CostAndCorrectAnswers& item);
	};

	/// <summary>
	/// Addition operator
	/// </summary>
	CostAndCorrectAnswers operator +(const CostAndCorrectAnswers& item1, const CostAndCorrectAnswers& item2);
}