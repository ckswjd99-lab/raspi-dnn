#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <iostream>
#include <fstream>
#include <cassert>

std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type);

std::vector<std::string> readLabels(const std::string& labelFilepath);
void printInferenceResults(const std::vector<float>& outputTensorValues, const std::vector<std::string>& labels, int batchSize);