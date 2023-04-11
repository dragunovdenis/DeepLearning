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

#include "LinAlg2d.h"
#include "../Utilities.h"

namespace DeepLearning
{
	template <class R>
	std::string Vector2d<R>::to_string() const
	{
		return std::string("{") + Utils::to_string(x) + ", " + Utils::to_string(y) + "}";
	}

	template <class R>
	bool Vector2d<R>::try_parse(const std::string& str, Vector2d<R>& out)
	{
		const auto scalars = Utils::parse_scalars<R>(str);

		if (scalars.size() == 1)
		{
			out.x = scalars[0];
			out.y = scalars[0];
			return true;
		}

		if (scalars.size() == 2)
		{
			out.x = scalars[0];
			out.y = scalars[1];
			return true;
		}

		return false;
	}

	template <class R>
	Vector2d<R> Vector2d<R>::random()
	{
		return { R(Utils::get_random(-1, 1)), R(Utils::get_random(-1, 1)) };
	}

	template<class R>
	Matrix2x2<R> Matrix2x2<R>::random()
	{
		return { R(Utils::get_random(-1, 1)),
				 R(Utils::get_random(-1, 1)),
				 R(Utils::get_random(-1, 1)),
				 R(Utils::get_random(-1, 1)), };
	}


	template std::string Vector2d<double>::to_string() const;
	template std::string Vector2d<float>::to_string() const;
	template std::string Vector2d<int>::to_string() const;
	template std::string Vector2d<long long int>::to_string() const;
	template std::string Vector2d<long unsigned>::to_string() const;

	template bool Vector2d<double>::try_parse(const std::string& str, Vector2d<double>& out);
	template bool Vector2d<float>::try_parse(const std::string& str, Vector2d<float>& out);
	template bool Vector2d<int>::try_parse(const std::string& str, Vector2d<int>& out);
	template bool Vector2d<long long int>::try_parse(const std::string& str, Vector2d<long long int>& out);
	template bool Vector2d<unsigned int>::try_parse(const std::string& str, Vector2d<unsigned int>& out);

	template Vector2d<float> Vector2d<float>::random();
	template Vector2d<double> Vector2d<double>::random();

	template Matrix2x2<double> Matrix2x2<double>::random();
	template Matrix2x2<float> Matrix2x2<float>::random();

}