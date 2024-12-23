#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <iostream>
#include <fstream>
#include <cassert>

#define PRT_COLOR_RED "\033[1;31m"
#define PRT_COLOR_GREEN "\033[1;32m"
#define PRT_COLOR_YELLOW "\033[1;33m"
#define PRT_COLOR_BLUE "\033[1;34m"
#define PRT_COLOR_MAGENTA "\033[1;35m"
#define PRT_COLOR_CYAN "\033[1;36m"
#define PRT_COLOR_RESET "\033[0m"

std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type);

std::vector<std::string> read_labels(const std::string& labelFilepath);
void print_inference_results(const std::vector<float>& outputTensorValues, const std::vector<std::string>& labels, int batchSize);