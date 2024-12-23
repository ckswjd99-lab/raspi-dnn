#include <onnxruntime/onnxruntime_cxx_api.h>

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
#include <pthread.h>

#include "util.hpp"
#include "scheduler.hpp"

#define CONFIG_PATH "./data/imnet.config"
#define IMAGE_PATH "./data/european-bee-eater-2115564_1920.jpg"
#define LABEL_PATH "./data/synset.txt"
#define NUM_TESTS 10
#define DEADLINE_MS 1000


int main(int argc, char* argv[])
{
    std::string config_filepath{CONFIG_PATH};
    std::string image_filepath{IMAGE_PATH};
    std::string label_filepath{LABEL_PATH};
    int num_tests = NUM_TESTS;

    const int64_t batch_size = 1;

    /* SCHEDULING */
    int deadline_ms = DEADLINE_MS;
    printf(PRT_COLOR_CYAN "Inference Scheduling\n" PRT_COLOR_RESET);
    printf("<Inference Information>\n");
    printf(" - Deadline: %d ms\n", deadline_ms);

    InferenceScheduler scheduler(label_filepath, DEFAULT_MAX_THREADS);
    scheduler.load_session_config(config_filepath);
    scheduler.load_input(image_filepath, batch_size);

    scheduler.benchmark(num_tests, 2);
    scheduler.reset_inference();

    scheduler.infer(std::chrono::system_clock::now() + std::chrono::milliseconds(deadline_ms));


    return 0;
}