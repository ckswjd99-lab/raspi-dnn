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

#include "infer_session.hpp"
#include "utils.hpp"

#define MODEL_RESNET18 "./data/resnet18-v1-7.onnx"   // 181 - 96 - 68 - 51 - 46 - 42 - 35 - 40 ms
#define MODEL_SQUEEZE  "./data/squeezenet1.1-7.onnx" // 36 - 20 - 15 - 12 - 11 - 9 - 9 - 11 ms

#define MODEL_PATH MODEL_SQUEEZE
#define IMAGE_PATH "./data/european-bee-eater-2115564_1920.jpg"
#define LABEL_PATH "./data/synset.txt"
#define NUM_TESTS 10


int main(int argc, char* argv[])
{
    std::string instance_name{"image-classification-inference"};
    std::string model_filepath{MODEL_PATH};
    std::string image_filepath{IMAGE_PATH};
    std::string label_filepath{LABEL_PATH};
    int num_intra_threads = 1;
    int num_inter_threads = 1;
    int num_multi_threads = 1;
    int num_tests = NUM_TESTS;

    const int64_t batch_size = 1;

    if (argc > 1) model_filepath = argv[1];
    if (argc > 2) image_filepath = argv[2];
    if (argc > 3) num_intra_threads = std::stoi(argv[3]);
    if (argc > 4) num_inter_threads = std::stoi(argv[4]);
    if (argc > 5) num_multi_threads = std::stoi(argv[5]);
    if (argc > 6) num_tests = std::stoi(argv[6]);
    if (argc > 7) {
        printf("Usage: %s [model] [image] [num_intra_threads] [num_inter_threads]\n", argv[0]);
        return 1;
    }

    /* SINGLE SESSION INFERENCE */
    printf(PRT_COLOR_CYAN "Single Session Inference\n" PRT_COLOR_RESET);

    InferenceSession session(instance_name, model_filepath, label_filepath, num_intra_threads, num_inter_threads);
    session.print_info();

    session.load_input(image_filepath, batch_size);

    std::vector<int64_t> elapsed_times;
    for (int i = 0; i < num_tests; i++) {
        auto begin = std::chrono::steady_clock::now();

        session.infer_sync();

        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        elapsed_times.push_back(elapsed);
        printf("Elapsed time: %ld ms\n", elapsed);

    }
    session.print_results();
    

    float elapsed_single_avg = (float)std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0) / num_tests;
    std::cout << "Average elapsed time: " << elapsed_single_avg << " ms" << std::endl;
    std::cout << std::endl;


    /* MULTI SESSION INFERENCE */
    printf(PRT_COLOR_CYAN "Multi Session Inference\n" PRT_COLOR_RESET);
    printf("<Inference Information>\n");
    printf(" - Number of sessions: %d\n", num_multi_threads);
    printf("\n");

    std::vector<InferenceSession*> sessions;
    for (int i = 0; i < num_multi_threads; i++) {
        sessions.push_back(new InferenceSession(instance_name, model_filepath, label_filepath, num_intra_threads, num_inter_threads));
        sessions[i]->load_input(image_filepath, batch_size);
    }

    std::vector<int64_t> elapsed_total;
    for (int i = 0; i < num_tests; i++) {
        auto begin = std::chrono::steady_clock::now();

        for (int j = 0; j < num_multi_threads; j++) {
            sessions[j]->infer_async();
        }
        for (int j = 0; j < num_multi_threads; j++) {
            sessions[j]->wait_infer();
        }

        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        elapsed_total.push_back(elapsed);

        printf("Elapsed time: %ld ms\n", elapsed_total.back());
    }
    sessions[0]->print_results();

    float elapsed_multi_avg = (float)std::accumulate(elapsed_total.begin(), elapsed_total.end(), 0) / num_tests;
    std::cout << "Average elapsed time: " << elapsed_multi_avg << " ms" << std::endl;
    std::cout << std::endl;
    

    return 0;
}