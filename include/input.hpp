#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


cv::Mat preprocess_image(const std::string& image_filepath, const std::vector<int64_t>& input_dims);
void prepareInputTensor(const std::string& image_filepath, const std::vector<int64_t>& input_dims, std::vector<float>& input_tensor_values, int64_t batch_size, size_t input_tensor_size);
