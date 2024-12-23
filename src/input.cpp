#include "input.hpp"


cv::Mat preprocess_image(const std::string& image_filepath, const std::vector<int64_t>& input_dims)
{
    cv::Mat image_BGR = cv::imread(image_filepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resized_image_BGR, resized_image_RGB, resized_image, preprocessed_image;
    cv::resize(image_BGR, resized_image_BGR, cv::Size(input_dims.at(3), input_dims.at(2)), cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resized_image_BGR, resized_image_RGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
    resized_image_RGB.convertTo(resized_image, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(resized_image, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resized_image);
    cv::dnn::blobFromImage(resized_image, preprocessed_image);

    return preprocessed_image;
}

void prepareInputTensor(const std::string& image_filepath, const std::vector<int64_t>& input_dims, std::vector<float>& input_tensor_values, int64_t batch_size, size_t input_tensor_size)
{
    cv::Mat preprocessed_image = preprocess_image(image_filepath, input_dims);
    
    for (int64_t i = 0; i < batch_size; ++i)
    {
        std::copy(preprocessed_image.begin<float>(), preprocessed_image.end<float>(), input_tensor_values.begin() + i * input_tensor_size / batch_size);
    }
}
