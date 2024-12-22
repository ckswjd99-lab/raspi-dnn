#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


cv::Mat preprocessImage(const std::string& imageFilepath, const std::vector<int64_t>& inputDims);
void copyImageToTensor(const cv::Mat& preprocessedImage, std::vector<float>& inputTensorValues, int64_t batchSize, size_t inputTensorSize);
void prepareInputTensor(const std::string& imageFilepath, const std::vector<int64_t>& inputDims, std::vector<float>& inputTensorValues, int64_t batchSize, size_t inputTensorSize);

void loadInputTensor(
    Ort::Session& session, const std::string& imageFilepath,
    int64_t batchSize, const std::vector<std::string>& labels,
    std::vector<const char*>& inputNames, std::vector<Ort::Value>& inputTensors,
    std::vector<const char*>& outputNames, std::vector<Ort::Value>& outputTensors
);