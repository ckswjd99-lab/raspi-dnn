#include "scheduler.hpp"
#include "session.hpp"
#include "util.hpp"

InferenceScheduler::InferenceScheduler(const std::string& label_path, int max_threads) {
    this->label_path = label_path;
    this->labels = read_labels(label_path);
    
    this->max_threads = max_threads;
    this->threads_using = 0;

    any_finished_mutex = PTHREAD_MUTEX_INITIALIZER;
    any_finished_cond = PTHREAD_COND_INITIALIZER;
}

InferenceScheduler::~InferenceScheduler() {
    for (auto session : sessions) {
        delete session;
    }
}

void InferenceScheduler::add_session(
    const std::string& model_path, float weight,
    int num_intra_threads, int num_inter_threads
) {
    std::string instance_name = std::to_string(sessions.size()) + "_" + model_path;
    InferenceSession* session = new InferenceSession(
        instance_name, model_path, label_path, 
        num_intra_threads, num_inter_threads
    );
    sessions.push_back(session);
    session_weights.push_back(0.0);
    session_inference_times.push_back(0.0);

    session->add_finish_listener(&any_finished_mutex, &any_finished_cond);
}

void InferenceScheduler::load_session_config(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(config_file, line)) {
        std::istringstream iss(line);
        std::string model_path;
        float weight;
        int num_intra_threads, num_inter_threads;
        iss >> model_path >> weight >> num_intra_threads >> num_inter_threads;
        add_session(model_path, weight, num_intra_threads, num_inter_threads);
    }
}

void InferenceScheduler::load_input(const std::string& image_path, int batch_size) {
    for (auto session : sessions) {
        session->load_input(image_path, batch_size);
    }
}

void InferenceScheduler::print_results() {
    for (auto session : sessions) {
        printf("<Instance Name: %s>\n", session->get_instance_name().c_str());
        session->print_results();
    }
}

void InferenceScheduler::benchmark(int num_runs, int num_warmup_runs) {
    PRINT_THREAD_MAIN("Benchmarking sessions");

    for (int snum = 0; snum < sessions.size(); snum++) {
        InferenceSession* session = sessions[snum];
        for (int i = 0; i < num_warmup_runs; i++) {
            session->infer_sync();
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; i++) {
            session->infer_sync();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        session_inference_times[snum] = duration / num_runs;

        PRINT_THREAD_MAIN(session->get_instance_name() << " (" << session_inference_times[snum] << " ms)");
    }
}

void InferenceScheduler::infer(std::chrono::_V2::system_clock::time_point deadline) {
    auto start = std::chrono::system_clock::now();
    int64_t start_ms = start.time_since_epoch().count() / 1000000;
    // not duration, a time point
    int64_t deadline_ms = deadline.time_since_epoch().count() / 1000000;

    while (true) {
        if (session_ready_queue.empty() && session_inference_queue.empty()) {
            PRINT_THREAD_MAIN("All sessions finished");
            break;
        }

        // check if the session can be started
        // 1) exist check: if there's no session in the ready queue, cannot start
        // 1) thread check: if the session starts, the number of using threads should be less than max_threads
        // 2) deadline check: if the session ends, the expected end time should be less than the deadline
        int flag_start = 1;
        int session_idx = -1;
        int session_num_threads = 0;
        InferenceSession* session = nullptr;

        if (session_ready_queue.empty()) {
            PRINT_THREAD_MAIN("No session in the ready queue");
            flag_start = 0;
        }
        else {
            session_idx = session_ready_queue.front();
            session = sessions[session_idx];
            PRINT_THREAD_MAIN("Checking session: " << session->get_instance_name());

            session_num_threads = session->get_num_intra_threads() * session->get_num_inter_threads();
            if (threads_using + session_num_threads > max_threads) {
                PRINT_THREAD_MAIN("May exceed thread limit: " << threads_using + session_num_threads << " > " << max_threads);
                flag_start = 0;
            }

            auto now = std::chrono::system_clock::now();
            int64_t now_ms = now.time_since_epoch().count() / 1000000;
            int64_t elapsed_ms = now_ms - start_ms;
            int64_t expected_latency_ms = session_inference_times[session_idx];
            int64_t expected_end_time_ms = elapsed_ms + expected_latency_ms;
            PRINT_THREAD_MAIN(
                "Elapsed " << elapsed_ms << " ms, " << 
                "Expected latency " << expected_latency_ms << " ms, " <<
                "Expected end time " << (elapsed_ms + expected_latency_ms) << " ms"
            );
            if (expected_end_time_ms > deadline_ms - start_ms) {
                PRINT_THREAD_MAIN("May exceed deadline: " << expected_end_time_ms << " > " << deadline_ms - start_ms);
                flag_start = 0;
            }
        }

        if (flag_start == 1) {
            PRINT_THREAD_MAIN("Starting session: " << session->get_instance_name());

            // start session
            threads_using += session_num_threads;
            session->infer_async();

            session_ready_queue.erase(session_ready_queue.begin());
            session_inference_queue.push_back(session_idx);
        }
        else {
            if (session != nullptr) {
                PRINT_THREAD_MAIN("Cannot start session: " << session->get_instance_name());
            }

            // wait for any session to finish
            pthread_mutex_lock(&any_finished_mutex);
            struct timespec deadline_as_timespec = timepoint_to_timespec(deadline);
            int ret = pthread_cond_timedwait(&any_finished_cond, &any_finished_mutex, &deadline_as_timespec);
            pthread_mutex_unlock(&any_finished_mutex);
            
            if (ret == 0) {
                PRINT_THREAD_MAIN("Any session finished");
            }

            // check finished sessions and update threads_using
            int session_iter = 0;
            while (session_inference_queue.size() > 0) {
                int session_idx = session_inference_queue[session_iter];
                InferenceSession* session = sessions[session_idx];
                if (session->get_state() == SESSION_STATE_FINISHED) {
                    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(session->get_finish_time() - start).count();
                    PRINT_THREAD_MAIN("Session finished: " << session->get_instance_name() << " (" << latency << " ms)");

                    threads_using -= session->get_num_intra_threads() * session->get_num_inter_threads();
                    session_inference_queue.erase(session_inference_queue.begin());
                    session_finished_queue.push_back(session_idx);
                }
                else {
                    session_iter++;
                    if (session_iter >= session_inference_queue.size()) {
                        break;
                    }
                }
            }

            // check if timeout
            if (ret == ETIMEDOUT) {
                PRINT_THREAD_MAIN("Deadline exceeded");

                // kill all alive threads
                for (auto session_idx : session_inference_queue) {
                    InferenceSession* session = sessions[session_idx];
                    pthread_cancel(session->get_thread());
                }

                break;
            }
        }
    }

    auto end = std::chrono::system_clock::now();
    int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    printf("Elapsed time: %ld ms\n", elapsed);
    printf("Finished sessions:\n");
    for (auto session_idx : session_finished_queue) {
        auto finish_time = sessions[session_idx]->get_finish_time();
        int64_t latency = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start).count();
        printf("\t%s (%ld ms)\n", sessions[session_idx]->get_instance_name().c_str(), latency);
    }
}

void InferenceScheduler::reset_inference() {
    for (auto session : sessions) {
        session->reset_state();
    }

    session_ready_queue.clear();
    session_inference_queue.clear();
    session_finished_queue.clear();
    enqueue_inference_naive();
}

void InferenceScheduler::enqueue_inference_naive() {
    for (int i = 0; i < sessions.size(); i++) {
        session_ready_queue.push_back(i);
    }
}