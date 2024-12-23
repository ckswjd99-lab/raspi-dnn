#pragma once

#include <iostream>
#include <string>


class InferenceScheduler {
    public:
    InferenceScheduler(const std::string& label_path);
    ~InferenceScheduler();

    void add_session(const std::string& model_path, int num_intra_threads, int num_inter_threads);


    private:
}