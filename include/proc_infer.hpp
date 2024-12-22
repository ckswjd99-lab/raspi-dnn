#pragma once

#include <iostream>
#include <vector>
#include <chrono>

#include <pthread.h>

#include <onnxruntime/onnxruntime_cxx_api.h>


class InferenceSession {
    public:
    InferenceSession(
        std::string instanceName,
        const std::string& model_path, const std::string& label_path,
        int num_intra_threads, int num_inter_threads
    );
    ~InferenceSession();

    void print_info();

    void load_input(const std::string& image_path, int batch_size);
    void print_results();
    
    void run();
    void wait_load();
    void infer();
    void wait_infer();
    void stop();


    private:
    std::string instanceName;
    std::string model_path;
    std::string label_path;
    int num_intra_threads;
    int num_inter_threads;
    std::vector<std::string> labels;
    
    Ort::Session* session = nullptr;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
    std::vector<float> inputTensorValues;
    std::vector<float> outputTensorValues;

    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;

    int run_flag = 0;
    int load_flag = 0;
    int infer_flag = 0;

    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    void *thread_func();


};