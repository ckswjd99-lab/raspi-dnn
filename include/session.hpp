#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>

#include <pthread.h>

#include <onnxruntime/onnxruntime_cxx_api.h>

#define SESSION_STATE_IDLE 0
#define SESSION_STATE_INFER 1
#define SESSION_STATE_FINISHED 2
#define SESSION_STATE_ZOMBIE 3


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

    void session_run();
    
    void infer_sync();
    int infer_async();
    void wait_infer();
    void add_finish_listener(pthread_mutex_t *mutex, pthread_cond_t *cond);
    void notify_finish_listeners();

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
    int64_t get_finish_time() { return finish_time_ts; }
    int get_num_inferenced() { return num_inferenced; }

    // setter functions
    void set_state(int state) { this->state = state; }
    void set_flag_infer(int flag_value) { atomic_store(&flag_infer, flag_value); }
    void set_finish_time(int64_t finish_time) { this->finish_time_ts = finish_time; }


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
    pthread_attr_t attr;
    std::vector<pthread_mutex_t *> finish_listeners_mutex;
    std::vector<pthread_cond_t *> finish_listeners_cond;
    int64_t finish_time_ts;

    int state;
    std::atomic_int flag_infer;     // indicates real state of inference
    int num_inferenced = 0;

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