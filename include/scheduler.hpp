#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <fstream>
#include <sstream>
#include <chrono>

#include "session.hpp"


class InferenceScheduler {
    public:
    InferenceScheduler(const std::string& label_path, int max_threads);
    ~InferenceScheduler();

    void add_session(
        const std::string& model_path, float weight,
        int num_intra_threads, int num_inter_threads
    );
    void load_session_config(const std::string& config_path);

    void load_input(const std::string& image_path, int batch_size);
    void print_results();

    void benchmark(int num_runs, int num_warmup_runs);
    void infer(std::chrono::_V2::system_clock::time_point deadline);

    void reset_inference();
    void enqueue_inference_naive();


    private:
    std::string label_path;
    std::vector<std::string> labels;

    int max_threads;
    int threads_using = 0;

    std::vector<InferenceSession*> sessions;
    std::vector<float> session_weights;
    std::vector<int64_t> session_inference_times;

    std::vector<int> session_ready_queue;
    std::vector<int> session_inference_queue;
    std::vector<int> session_finished_queue;

    pthread_mutex_t any_finished_mutex;
    pthread_cond_t any_finished_cond;

};