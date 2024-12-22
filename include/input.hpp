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
void prepareInputTensor(const std::string& imageFilepath, const std::vector<int64_t>& inputDims, std::vector<float>& inputTensorValues, int64_t batchSize, size_t inputTensorSize);
