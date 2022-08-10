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
#include <msgpack.hpp>
#include <sstream>
#include <fstream>
#include <filesystem>

namespace DeepLearning
{
	namespace MsgPack
	{
		/// <summary>
		/// Serializes the given object in message-pack format
		/// </summary>
		template <class T>
		std::stringstream pack(const T& obj)
		{
			std::stringstream result;
			msgpack::pack(result, obj);
			return result;
		}

		/// <summary>
		/// Serializes the given object in message-pack format and saves it to the given file on disk
		/// </summary>
		template <class T>
		void save_to_file(const T& obj, const std::filesystem::path& file_name)
		{
			std::ofstream file(file_name, std::ios::out | std::ios::binary);

			if (!file.is_open())
				throw std::exception("Can't create file");

			const auto message = pack(obj);
			file << message.rdbuf();
			file.close();
		}

		/// <summary>
		/// Deserializes an instance of class T from the given message
		/// </summary>
		template <class T> 
		T unpack(const std::stringstream& msg)
		{
			msgpack::unpacked result;
			msgpack::unpack(result, msg.str().data(), msg.str().size());
			return result.get().as<T>();
		}

		/// <summary>
		/// Tries to deserialize an instance of the given class T from the message-pack data loaded from the given file
		/// </summary>
		template <class T>
		T load_from_file(const std::filesystem::path& file_name)
		{
			std::ifstream file(file_name, std::ios::in | std::ios::binary);

			if (!file.is_open())
				throw std::exception("Can't open file");

			std::stringstream message;
			message << file.rdbuf();
			file.close();

			return unpack<T>(message);
		}
	}
}
