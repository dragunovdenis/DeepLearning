//Copyright (c) 2026 Denys Dragunov, dragunovdenis@gmail.com
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

#include "CppUnitTest.h"
#include <NeuralNet/Net.h>
#include <NeuralNet/DataContextCuda.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	//This test works but I am not satisfied with its performance (execution time), so it is out-commented for now
	//TEST_METHOD(CudaTrainingConvolutionNetWithCrossEntropyCostTest)
	//{
	//	constexpr bool long_test = false;
	//	RunMnistBasedConvolutionNetTrainingTest<GpuDC>(CostFunctionId::CROSS_ENTROPY, static_cast<Real>(0.03),
	//		long_test ? static_cast<Real>(0.991) : static_cast<Real>(0.98), long_test, static_cast<Real>(3));
	//}

	//This test works but I am not satisfied with its performance (execution time), so it is out-commented for now
	//TEST_METHOD(CudaTrainingWithCrossEntropyCostAndSoftMaxActivationTest)
	//{
	//	constexpr bool long_test = false;
	//	RunMnistBasedTrainingTest<GpuDC>(CostFunctionId::CROSS_ENTROPY, Real(0.1), long_test ? Real(0.978) : Real(0.95), long_test, Real(0),
	//		{ ActivationFunctionId::SIGMOID, ActivationFunctionId::SOFTMAX });
	//}

	/// <summary>
	/// Instantiate template with the GPU data context to ensure that it is compilable.
	/// </summary>
	template class DeepLearning::Net<GpuDC>;
}
