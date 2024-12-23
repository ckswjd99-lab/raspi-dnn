#pragma once

#include <iostream>
#include <vector>
#include <chrono>

#include <pthread.h>

#include <onnxruntime/onnxruntime_cxx_api.h>

#define SESSION_STATE_IDLE 0
#define SESSION_STATE_INFER 1
#define SESSION_STATE_FINISHED 2


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
    void add_finish_listener(pthread_mutex_t *mutex, pthread_cond_t *cond);

    void reset_state();

    // getter functions
    std::vector<float> get_output_tensor_values() { return output_tensor_values; }
    std::vector<std::string> get_labels() { return labels; }
    std::string get_instance_name() { return instance_name; }
    std::string get_model_path() { return model_path; }
    std::string get_label_path() { return label_path; }
    int get_num_intra_threads() { return num_intra_threads; }
    int get_num_inter_threads() { return num_inter_threads; }
    pthread_t get_thread() { return thread; }
    int get_state() { return state; }
    std::chrono::system_clock::time_point get_finish_time() { return finish_time; }


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
    std::vector<pthread_mutex_t *> finish_listeners_mutex;
    std::vector<pthread_cond_t *> finish_listeners_cond;
    std::chrono::system_clock::time_point finish_time;

    int flag_infer = 0;
    int state = SESSION_STATE_IDLE;

    void* infer_async_func(void* arg);

};

void test_single_session(
    const std::string& model_filepath, const std::string& label_filepath, const std::string& image_filepath,
    int num_intra_threads, int num_inter_threads,
    int batch_size, int num_tests
);

void test_multi_session(
    const std::string& model_filepath, const std::string& label_filepath, const std::string& image_filepath,
    int num_intra_threads, int num_inter_threads, int num_multi_threads,
    int batch_size, int num_tests
);