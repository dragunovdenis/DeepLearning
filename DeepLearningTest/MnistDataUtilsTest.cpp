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

#include "CppUnitTest.h"
#include <MnistDataUtils.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(MnistDataUtilsTest)
	{
		/// <summary>
		/// Validates given collection of MNIST labels
		/// </summary>
		static void ValidateLabels(const std::vector<DenseVector>& labels)
		{
			for (std::size_t label_id = 0; label_id < labels.size(); label_id++)
			{
				const auto& label = labels[label_id];
				Assert::IsTrue(label.dim() == 10, (std::wstring(L"Unexpected dimension of a label. Id = ") + std::to_wstring(label_id)).c_str());
				int number_of_nonzero_components = 0;
				for (std::size_t element_id = 0; element_id < label.dim(); element_id++)
					if (std::abs(label(element_id)) > 0)
						number_of_nonzero_components++;

				Assert::IsTrue(number_of_nonzero_components == 1,
					(std::wstring(L"Unexpected number of nonzero elements in a label. Id =") + std::to_wstring(label_id)).c_str());

				//check that the only nonzero element is equal to "1"
				Assert::IsTrue(label.max_abs() == Real(1),
					(std::wstring(L"Unexpected value of the nonzero element in a label. Id =") + std::to_wstring(label_id)).c_str());
			}
		}

	public:
		TEST_METHOD(TrainingImageReadingTest)
		{
			const auto training_images_count = 60000;
			const auto images = MnistDataUtils::read_images("TestData\\MNIST\\train-images.idx3-ubyte", training_images_count);
			Assert::AreEqual<std::size_t>(images.size(), training_images_count, L"Unexpected number of training images");

			return;
			for (std::size_t image_id = 0; image_id < 0.01*images.size(); image_id++)
			{
				const auto& image = images[image_id];
				image.SaveToBmp(std::string("D:\\Development\\SandBox\\DeepLearning\\image_") + std::to_string(image_id) + ".bmp");
			}
		}

		TEST_METHOD(TrainingLabelsReadingTest)
		{
			const auto training_labels_count = 60000;
			const auto labels = MnistDataUtils::read_labels("TestData\\MNIST\\train-labels.idx1-ubyte", training_labels_count);
			Assert::AreEqual<std::size_t>(labels.size(), training_labels_count, L"Unexpected number of training labels");
			ValidateLabels(labels);
		}

		TEST_METHOD(TestImageReadingTest)
		{
			const auto test_images_count = 10000;
			const auto images = MnistDataUtils::read_images("TestData\\MNIST\\t10k-images.idx3-ubyte", test_images_count);
			Assert::AreEqual<std::size_t>(images.size(), test_images_count, L"Unexpected number of test images");
		}

		TEST_METHOD(TestLabelsReadingTest)
		{
			const auto training_labels_count = 10000;
			const auto labels = MnistDataUtils::read_labels("TestData\\MNIST\\t10k-labels.idx1-ubyte", training_labels_count);
			Assert::AreEqual<std::size_t>(labels.size(), training_labels_count, L"Unexpected number of test labels");
			ValidateLabels(labels);
		}
	};
}
