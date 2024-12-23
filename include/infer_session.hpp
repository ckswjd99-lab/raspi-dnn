#pragma once

#include <iostream>
#include <vector>
#include <chrono>

#include <pthread.h>

#include <onnxruntime/onnxruntime_cxx_api.h>


class InferenceSession {
    public:
    InferenceSession(
        std::string instance_name,
        const std::string& model_path, const std::string& label_path,
        int num_intra_threads, int num_inter_threads
    );
    ~InferenceSession();

    void print_info();

    void load_input(const std::string& image_path, int batch_size);
    void print_results();
    
    void infer_sync();
    void infer_async();
    void wait_infer();


    private:
    std::string instance_name;
    std::string model_path;
    std::string label_path;
    int num_intra_threads;
    int num_inter_threads;
    std::vector<std::string> labels;
    
    Ort::Session* session = nullptr;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    std::vector<float> input_tensor_values;
    std::vector<float> output_tensor_values;

    std::vector<Ort::AllocatedStringPtr> input_node_name_allocated_strings;
    std::vector<Ort::AllocatedStringPtr> output_node_name_allocated_strings;

    pthread_t thread;

    int flag_infer = 0;

};