#include "CppUnitTest.h"
#include <MnistDataUtils.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DeepLearning;

namespace DeepLearningTest
{
	TEST_CLASS(MnistDataUtilsTest)
	{
	public:
		TEST_METHOD(TrainingImageReadingTest)
		{
			const auto training_images_count = 60000;
			const auto images = MnistDataUtils::read_images("TestData\\MNIST\\train-images.idx3-ubyte", training_images_count);
			Assert::AreEqual<std::size_t>(images.size(), training_images_count, L"Unexpected number of training images");
		}

		TEST_METHOD(TestImageReadingTest)
		{
			const auto test_images_count = 10000;
			const auto images = MnistDataUtils::read_images("TestData\\MNIST\\t10k-images.idx3-ubyte", test_images_count);
			Assert::AreEqual<std::size_t>(images.size(), test_images_count, L"Unexpected number of test images");
		}
	};
}
