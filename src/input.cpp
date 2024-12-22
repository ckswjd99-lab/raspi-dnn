#include "input.hpp"


cv::Mat preprocessImage(const std::string& imageFilepath, const std::vector<int64_t>& inputDims)
{
    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR, cv::Size(inputDims.at(3), inputDims.at(2)), cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    return preprocessedImage;
}

void prepareInputTensor(const std::string& imageFilepath, const std::vector<int64_t>& inputDims, std::vector<float>& inputTensorValues, int64_t batchSize, size_t inputTensorSize)
{
    cv::Mat preprocessedImage = preprocessImage(imageFilepath, inputDims);
    
    for (int64_t i = 0; i < batchSize; ++i)
    {
        std::copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), inputTensorValues.begin() + i * inputTensorSize / batchSize);
    }
}
