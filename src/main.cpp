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
#include <thread>

#include "util.hpp"
#include "scheduler.hpp"

#define CONFIG_PATH "./data/imnet.config"
#define IMAGE_PATH "./data/european-bee-eater-2115564_1920.jpg"
#define LABEL_PATH "./data/synset.txt"
#define DEADLINE_MS 33
#define NUM_TESTS 30


int main(int argc, char* argv[])
{
    std::string config_filepath{CONFIG_PATH};
    std::string image_filepath{IMAGE_PATH};
    std::string label_filepath{LABEL_PATH};
    int deadline_ms = DEADLINE_MS;
    int num_tests = NUM_TESTS;

    const int64_t batch_size = 1;

    // Parse arguments
    if (argc > 1) { config_filepath = argv[1]; }
    if (argc > 2) { deadline_ms = std::stoi(argv[2]); }
    if (argc > 3) { num_tests = std::stoi(argv[3]); }

    /* SCHEDULING */
    printf(PRT_COLOR_CYAN "Inference Scheduling\n" PRT_COLOR_RESET);
    printf("<Inference Information>\n");
    printf(" - Deadline: %d ms\n", deadline_ms);

    InferenceScheduler scheduler(label_filepath, DEFAULT_MAX_THREADS);
    scheduler.load_session_config(config_filepath);
    scheduler.load_input(image_filepath, batch_size);

    scheduler.benchmark(num_tests, 2);

    auto start = std::chrono::system_clock::now();
    std::vector<int64_t> elapsed_times;
    for (int i = 0; i < num_tests; i++)
    {
        scheduler.reset_inference();
        
        // wait until start + deadline * i
        auto target_time = start + std::chrono::milliseconds(deadline_ms * i);
        std::this_thread::sleep_until(target_time);

        scheduler.infer(get_current_time_milliseconds() + deadline_ms);

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
        elapsed_times.push_back(elapsed);
    }

    for (auto elapsed_ms : elapsed_times)
    {
        printf("Elapsed time: %lld ms\n", elapsed_ms);
    }

    return 0;
}