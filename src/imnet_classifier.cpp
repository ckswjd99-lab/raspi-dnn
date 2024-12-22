// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <pthread.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <cassert>

#include "utils.hpp"

#define MODEL_RESNET18 "./data/resnet18-v1-7.onnx"   // 181 - 96 - 68 - 51 - 46 - 42 - 35 - 40 ms
#define MODEL_SQUEEZE  "./data/squeezenet1.1-7.onnx" // 36 - 20 - 15 - 12 - 11 - 9 - 9 - 11 ms

#define MODEL_PATH MODEL_SQUEEZE
#define IMAGE_PATH "./data/european-bee-eater-2115564_1920.jpg"
#define LABEL_PATH "./data/synset.txt"
#define NUM_TESTS 10

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
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

void printInferenceResults(const std::vector<float>& outputTensorValues, const std::vector<std::string>& labels, int batchSize)
{
    std::vector<int> predIds(batchSize, 0);
    std::vector<std::string> predLabels(batchSize);
    std::vector<float> confidences(batchSize, 0.0f);
    for (int64_t b = 0; b < batchSize; ++b)
    {
        float activation = 0;
        float maxActivation = std::numeric_limits<float>::lowest();
        float expSum = 0;
        for (int i = 0; i < labels.size(); i++)
        {
            activation = outputTensorValues.at(i + b * labels.size());
            expSum += std::exp(activation);
            if (activation > maxActivation)
            {
                predIds.at(b) = i;
                maxActivation = activation;
            }
        }
        predLabels.at(b) = labels.at(predIds.at(b));
        confidences.at(b) = std::exp(maxActivation) / expSum;
    }
    for (int64_t b = 0; b < batchSize; ++b)
    {
        assert(("Output predictions should all be identical.", predIds.at(b) == predIds.at(0)));
    }

    std::cout << "Predicted Label ID: " << predIds.at(0) << std::endl;
    std::cout << "Predicted Label: " << predLabels.at(0) << std::endl;
    std::cout << "Uncalibrated Confidence: " << confidences.at(0) << std::endl;
}

void copyImageToTensor(const cv::Mat& preprocessedImage, std::vector<float>& inputTensorValues, int64_t batchSize, size_t inputTensorSize)
{
    for (int64_t i = 0; i < batchSize; ++i)
    {
        std::copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), inputTensorValues.begin() + i * inputTensorSize / batchSize);
    }
}

Ort::Session createSession(const std::string& modelFilepath, const std::string& instanceName, int numIntraThreads, int numInterThreads)
{
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(numIntraThreads);
    sessionOptions.SetInterOpNumThreads(numInterThreads);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    return Ort::Session(env, modelFilepath.c_str(), sessionOptions);
}

struct ThreadData {
    Ort::Session* session;
    const std::vector<const char*>* inputNames;
    const std::vector<Ort::Value>* inputTensors;
    const std::vector<const char*>* outputNames;
    std::vector<Ort::Value>* outputTensors;
    std::chrono::steady_clock::time_point end;
};

void* runInferenceThread(void* arg)
{
    ThreadData* data = static_cast<ThreadData*>(arg);
    data->session->Run(
        Ort::RunOptions{nullptr}, 
        data->inputNames->data(), data->inputTensors->data(), 1, 
        data->outputNames->data(), data->outputTensors->data(), 1
    );
    data->end = std::chrono::steady_clock::now();
    return nullptr;
}

void prepareInputTensor(const std::string& imageFilepath, const std::vector<int64_t>& inputDims, std::vector<float>& inputTensorValues, int64_t batchSize, size_t inputTensorSize)
{
    cv::Mat preprocessedImage = preprocessImage(imageFilepath, inputDims);
    copyImageToTensor(preprocessedImage, inputTensorValues, batchSize, inputTensorSize);
}

int main(int argc, char* argv[])
{
    std::string instanceName{"image-classification-inference"};
    std::string modelFilepath{MODEL_PATH};
    std::string imageFilepath{IMAGE_PATH};
    std::string labelFilepath{LABEL_PATH};
    int numIntraThreads = 1;
    int numInterThreads = 1;
    int numMultiThreads = 1;
    int numTests = NUM_TESTS;

    const int64_t batchSize = 1;

    if (argc > 1) modelFilepath = argv[1];
    if (argc > 2) imageFilepath = argv[2];
    if (argc > 3) numIntraThreads = std::stoi(argv[3]);
    if (argc > 4) numInterThreads = std::stoi(argv[4]);
    if (argc > 5) numMultiThreads = std::stoi(argv[5]);
    if (argc > 6) numTests = std::stoi(argv[6]);
    if (argc > 7) {
        printf("Usage: %s [model] [image] [num_intra_threads] [num_inter_threads]\n", argv[0]);
        return 1;
    }

    printf("<Configuration>\n");
    std::cout << "Model: " << modelFilepath << std::endl;
    std::cout << "Image: " << imageFilepath << std::endl;
    printf("Num of Threads (intra, inter): (%d, %d)\n", numIntraThreads, numInterThreads);
    printf("Num of Threads (multi): %d\n", numMultiThreads);

    std::vector<std::string> labels{readLabels(labelFilepath)};

    std::vector<Ort::Session> sessionList;
    for (int i = 0; i < numMultiThreads; i++)
    {
        sessionList.push_back(createSession(modelFilepath, instanceName, numIntraThreads, numInterThreads));
    }

    Ort::Session& session = sessionList.at(0);
    
    
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

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

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
    std::cout << "Input Name: " << inputNames[0] << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Dimensions: " << inputDims << std::endl;
    std::cout << "Output Name: " << outputNames[0] << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize, outputDims.data(), outputDims.size()));

    
    // Single Thread Inference
    printf("\n");
    printf("<TEST: Single DNN>\n");
    
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < numTests; i++)
    {
        session.Run(
            Ort::RunOptions{nullptr}, 
            inputNames.data(), inputTensors.data(), 1, 
            outputNames.data(), outputTensors.data(), 1
        );
    }
    auto end = std::chrono::steady_clock::now();

    printInferenceResults(outputTensorValues, labels, batchSize);
    std::cout << "Inference Latency: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / static_cast<float>(numTests) << " ms" << std::endl;

    // Multi Thread Inference
    printf("\n");
    printf("<TEST: Multi DNN>\n");

    for (int i = 0; i < numTests; i++)
    {
        printf("Test %d\n", i);

        auto begin = std::chrono::steady_clock::now();
        pthread_t threads[numMultiThreads];
        ThreadData threadData[numMultiThreads];
        for (int i = 0; i < numMultiThreads; i++)
        {
            threadData[i] = {&sessionList.at(i), &inputNames, &inputTensors, &outputNames, &outputTensors};
            pthread_create(&threads[i], nullptr, runInferenceThread, &threadData[i]);
        }
        for (int i = 0; i < numMultiThreads; i++)
        {
            pthread_join(threads[i], nullptr);
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << "Total Inference Latency: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

        for (int i = 0; i < numMultiThreads; i++)
        {
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(threadData[i].end - begin).count() << " / ";
        }
        std::cout << std::endl;
    }
}