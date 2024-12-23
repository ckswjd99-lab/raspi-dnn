#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <iostream>
#include <fstream>
#include <cassert>
#include <ctime>
#include <chrono>

#define PRT_COLOR_RED "\033[1;31m"
#define PRT_COLOR_GREEN "\033[1;32m"
#define PRT_COLOR_YELLOW "\033[1;33m"
#define PRT_COLOR_BLUE "\033[1;34m"
#define PRT_COLOR_MAGENTA "\033[1;35m"
#define PRT_COLOR_CYAN "\033[1;36m"
#define PRT_COLOR_RESET "\033[0m"

#define PREFIX_THREAD_SUB "\033[1;33m[ST]\033[0m "
#define PREFIX_THREAD_MAIN "\033[1;32m[MT]\033[0m "
#ifdef DEBUG_THREAD
#define PRINT_THREAD_SUB(msg) std::cout << PREFIX_THREAD_SUB << msg << std::endl
#define PRINT_THREAD_MAIN(msg) std::cout << PREFIX_THREAD_MAIN << msg << std::endl
#else
#define PRINT_THREAD_SUB(msg)
#define PRINT_THREAD_MAIN(msg)
#endif

#define DEFAULT_MAX_THREADS 8

std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type);

std::vector<std::string> read_labels(const std::string& labelFilepath);
void print_inference_results(const std::vector<float>& outputTensorValues, const std::vector<std::string>& labels, int batchSize);

struct timespec timepoint_to_timespec(std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> tp);
