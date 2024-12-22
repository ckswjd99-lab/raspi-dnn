#include "proc_input.hpp"

template <typename T>
static T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
static std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

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

void copyImageToTensor(const cv::Mat& preprocessedImage, std::vector<float>& inputTensorValues, int64_t batchSize, size_t inputTensorSize)
{
    for (int64_t i = 0; i < batchSize; ++i)
    {
        std::copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), inputTensorValues.begin() + i * inputTensorSize / batchSize);
    }
}

void prepareInputTensor(const std::string& imageFilepath, const std::vector<int64_t>& inputDims, std::vector<float>& inputTensorValues, int64_t batchSize, size_t inputTensorSize)
{
    cv::Mat preprocessedImage = preprocessImage(imageFilepath, inputDims);
    copyImageToTensor(preprocessedImage, inputTensorValues, batchSize, inputTensorSize);
}

void loadInputTensor(
    Ort::Session& session, const std::string& imageFilepath,
    int64_t batchSize, const std::vector<std::string>& labels,
    std::vector<const char*>& inputNames, std::vector<Ort::Value>& inputTensors,
    std::vector<const char*>& outputNames, std::vector<Ort::Value>& outputTensors
) {
    std::cout << "<Load Input Tensor>" << std::endl;
    std::cout << &inputNames << std::endl;
    std::cout << &outputNames << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting input batch size to " << batchSize << "." << std::endl;
        inputDims.at(0) = batchSize;
    }

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting output batch size to " << batchSize << "." << std::endl;
        outputDims.at(0) = batchSize;
    }

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    prepareInputTensor(imageFilepath, inputDims, inputTensorValues, batchSize, inputTensorSize);

    size_t outputTensorSize = vectorProduct(outputDims);
    assert(("Output tensor size should equal to the label set size.", labels.size() * batchSize == outputTensorSize));
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;

    auto inputNodesNum = session.GetInputCount();
    for (int i = 0; i < inputNodesNum; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        inputNames.push_back(inputNodeNameAllocatedStrings.back().get());
    }

    auto outputNodesNum = session.GetOutputCount();
    for (int i = 0; i < outputNodesNum; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
    }

    std::cout << std::endl;
    std::cout << "<Model Information>" << std::endl;
    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Input Name: " << *inputNames.data() << inputNames.data() << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Dimensions: " << inputDims << std::endl;
    std::cout << "Output Name: " << *outputNames.data() << outputNames.data() << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize, outputDims.data(), outputDims.size()));

}