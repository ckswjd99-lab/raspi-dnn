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

#define MODEL_RESNET18 "./data/resnet18-v1-7.onnx"   // 181 - 96 - 68 - 51 - 46 - 42 - 35 - 40 ms
#define MODEL_SQUEEZE  "./data/squeezenet1.1-7.onnx" // 36 - 20 - 15 - 12 - 11 - 9 - 9 - 11 ms

#define MODEL_PATH MODEL_SQUEEZE
#define IMAGE_PATH "./data/european-bee-eater-2115564_1920.jpg"
#define LABEL_PATH "./data/synset.txt"
#define NUM_TESTS 10


int main(int argc, char* argv[])
{
    std::string instanceName{"image-classification-inference"};
    std::string modelFilepath{MODEL_PATH};
    std::string imageFilepath{IMAGE_PATH};
    std::string labelFilepath{LABEL_PATH};
    int numIntraThreads = 1;
    int numInterThreads = 1;
    int numMultiThreads = 1;
    int numTests = NUM_TESTS;

    const int64_t batchSize = 1;

    if (argc > 1) modelFilepath = argv[1];
    if (argc > 2) imageFilepath = argv[2];
    if (argc > 3) numIntraThreads = std::stoi(argv[3]);
    if (argc > 4) numInterThreads = std::stoi(argv[4]);
    if (argc > 5) numMultiThreads = std::stoi(argv[5]);
    if (argc > 6) numTests = std::stoi(argv[6]);
    if (argc > 7) {
        printf("Usage: %s [model] [image] [num_intra_threads] [num_inter_threads]\n", argv[0]);
        return 1;
    }

    InferenceSession session(instanceName, modelFilepath, labelFilepath, numIntraThreads, numInterThreads);
    session.print_info();

    session.load_input(imageFilepath, batchSize);

    session.run();
    session.wait_load();

    for (int i = 0; i < numTests; i++) {
        session.infer();
        session.wait_infer();
    }
    
    session.stop();


    return 0;
}