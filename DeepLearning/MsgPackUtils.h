#pragma once
#include <msgpack.hpp>
#include <sstream>

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

		template <class T> 
		T unpack(const std::stringstream& msg)
		{
			msgpack::unpacked result;
			msgpack::unpack(result, msg.str().data(), msg.str().size());
			return result.get().as<T>();
		}
	}
}
