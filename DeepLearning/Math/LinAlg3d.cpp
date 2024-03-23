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

#include "LinAlg3d.h"
#include "../Utilities.h"

namespace DeepLearning
{
	template <class T>
	Vector3d<T> Vector3d<T>::random(const T& min, const T& max)
	{
		return { T(Utils::get_random(static_cast<Real>(min), static_cast<Real>(max))),
			T(Utils::get_random(static_cast<Real>(min), static_cast<Real>(max))),
			T(Utils::get_random(static_cast<Real>(min), static_cast<Real>(max))) };
	}

	template <class T>
	std::string Vector3d<T>::to_string() const
	{
		return std::string("{") + Utils::to_string(x) + ", " + Utils::to_string(y) + ", " + Utils::to_string(z) + "}";
	}

	template <class T>
	bool Vector3d<T>::try_parse(const std::string& str, Vector3d<T>& out)
	{
		const auto scalars = Utils::parse_scalars<T>(str);

		if (scalars.size() == 1)
		{
			out.x = scalars[0];
			out.y = scalars[0];
			out.z = scalars[0];
			return true;
		}

		if (scalars.size() == 3)
		{
			out.x = scalars[0];
			out.y = scalars[1];
			out.z = scalars[2];
			return true;
		}

		return false;
	}

	template Vector3d<double> Vector3d<double>::random(const double& min, const double& max);
	template Vector3d<float> Vector3d<float>::random(const float& min, const float& max);

	template std::string Vector3d<double>::to_string() const;
	template std::string Vector3d<float>::to_string() const;
	template std::string Vector3d<int>::to_string() const;
	template std::string Vector3d<long long int>::to_string() const;
	template std::string Vector3d<unsigned int>::to_string() const;

	template bool Vector3d<double>::try_parse(const std::string& str, Vector3d<double>& out);
	template bool Vector3d<float>::try_parse(const std::string& str, Vector3d<float>& out);
	template bool Vector3d<int>::try_parse(const std::string& str, Vector3d<int>& out);
	template bool Vector3d<long long int>::try_parse(const std::string& str, Vector3d<long long int>& out);
	template bool Vector3d<unsigned int>::try_parse(const std::string& str, Vector3d<unsigned int>& out);
}