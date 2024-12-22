#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <iostream>
#include <fstream>

std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type);
std::vector<std::string> readLabels(const std::string& labelFilepath);