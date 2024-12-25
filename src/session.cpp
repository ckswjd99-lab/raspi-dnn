#include "session.hpp"
#include "util.hpp"
#include "input.hpp"


template <typename T>
static T vector_product(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
static std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

Ort::Session *create_session(const std::string& model_filepath, const std::string& instance_name, int num_intra_threads, int num_inter_threads)
{
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instance_name.c_str());
    Ort::SessionOptions session_options;
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetIntraOpNumThreads(num_intra_threads);
    session_options.SetInterOpNumThreads(num_inter_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    return new Ort::Session(env, model_filepath.c_str(), session_options);
}

InferenceSession::InferenceSession(
    std::string instance_name,
    const std::string& model_path, const std::string& label_path,
    int num_intra_threads, int num_inter_threads
) : instance_name(instance_name), model_path(model_path), label_path(label_path), num_intra_threads(num_intra_threads), num_inter_threads(num_inter_threads)
{
    labels = read_labels(label_path);
    session = create_session(model_path, instance_name, num_intra_threads, num_inter_threads);
    state = SESSION_STATE_IDLE;
    std::atomic_store(&flag_infer, 0);
}

InferenceSession::~InferenceSession() {
    
}

void InferenceSession::print_info() {
    printf("<Model Information>\n");
    printf(" - Instance Name: %s\n", instance_name.c_str());
    printf(" - Model Path: %s\n", model_path.c_str());
    printf(" - Label Path: %s\n", label_path.c_str());
    printf(" - Number of (Intra, Inter) Threads: (%d, %d)\n", num_intra_threads, num_inter_threads);
    printf("\n");
}

void InferenceSession::load_input(const std::string& image_path, int batch_size)
{
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session->GetInputCount();
    size_t num_output_nodes = session->GetOutputCount();

    Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    if (input_dims.at(0) == -1)
    {
        input_dims.at(0) = batch_size;
    }

    Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = output_tensor_info.GetElementType();
    std::vector<int64_t> outputDims = output_tensor_info.GetShape();
    if (outputDims.at(0) == -1)
    {
        outputDims.at(0) = batch_size;
    }

    size_t inputTensorSize = vector_product(input_dims);
    input_tensor_values.resize(inputTensorSize);
    prepareInputTensor(image_path, input_dims, input_tensor_values, batch_size, inputTensorSize);

    size_t outputTensorSize = vector_product(outputDims);
    assert(("Output tensor size should equal to the label set size.", labels.size() * batch_size == outputTensorSize));
    output_tensor_values.resize(outputTensorSize);

    auto inputNodesNum = session->GetInputCount();
    for (int i = 0; i < inputNodesNum; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_node_name_allocated_strings.push_back(std::move(input_name));
        input_names.push_back(input_node_name_allocated_strings.back().get());
    }

    auto outputNodesNum = session->GetOutputCount();
    for (int i = 0; i < outputNodesNum; i++) {
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        output_node_name_allocated_strings.push_back(std::move(output_name));
        output_names.push_back(output_node_name_allocated_strings.back().get());
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, input_tensor_values.data(), inputTensorSize, input_dims.data(), input_dims.size()));
    output_tensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, output_tensor_values.data(), outputTensorSize, outputDims.data(), outputDims.size()));

}

void InferenceSession::print_results()
{
    output_tensor_values.assign(output_tensors.at(0).GetTensorMutableData<float>(), output_tensors.at(0).GetTensorMutableData<float>() + output_tensors.at(0).GetTensorTypeAndShapeInfo().GetElementCount());
    print_inference_results(output_tensor_values, labels, 1);
}

void InferenceSession::session_run()
{
    Ort::RunOptions run_options{nullptr};
    session->Run(run_options, input_names.data(), input_tensors.data(), 1, output_names.data(), output_tensors.data(), 1);
}

void InferenceSession::infer_sync()
{
    // state = SESSION_STATE_INFER;
    
    // test and set flag
    bool ret = std::atomic_exchange(&flag_infer, 1);
    if (ret) {
        PRINT_THREAD_MAIN("infer_sync() called while the session is already running.");
        return;
    }

    state = SESSION_STATE_INFER;

    session->Run(
        Ort::RunOptions{nullptr}, 
        input_names.data(), input_tensors.data(), 1, 
        output_names.data(), output_tensors.data(), 1
    );

    finish_time_ts = get_current_time_milliseconds();

    state = SESSION_STATE_FINISHED;
    std::atomic_store(&flag_infer, 0);
}

void *infer_async_func(void* arg)
{
    InferenceSession* session = (InferenceSession*)arg;

    int inference_id = session->get_num_inferenced();
    session->set_state(SESSION_STATE_INFER);

    PRINT_THREAD_SUB("Inference start: " << session->get_instance_name());

    session->session_run();

    int64_t finish_time_ts = get_current_time_milliseconds();
    session->set_finish_time(finish_time_ts);

    // Check inference validity
    if (inference_id != session->get_num_inferenced())
    {
        PRINT_THREAD_SUB("Inference canceled: " << session->get_instance_name());
        
        session->set_state(SESSION_STATE_ZOMBIE);
        session->set_flag_infer(0);
        return nullptr;
    }

    // Notify finish listeners
    session->set_state(SESSION_STATE_FINISHED);
    session->notify_finish_listeners();

    PRINT_THREAD_SUB("Inference end: " << session->get_instance_name());

    session->set_flag_infer(0);
    return nullptr;
}

int InferenceSession::infer_async()
{
    int ret = std::atomic_exchange(&flag_infer, 1);
    if (ret == 1) {
        PRINT_THREAD_MAIN("infer_async() called while infer_async() is already running.");
        return -1;
    }

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    ret = pthread_create(&thread, &attr, (void* (*)(void*))&infer_async_func, this);

    return ret;
}

void InferenceSession::wait_infer()
{
    int ret = std::atomic_load(&flag_infer);
    if (ret == 0) {
        PRINT_THREAD_MAIN("wait_infer() called while infer_async() is not running.");
        return;
    }

    while (std::atomic_load(&flag_infer) == 1) { }
}

void InferenceSession::add_finish_listener(pthread_mutex_t* mutex, pthread_cond_t* cond)
{
    finish_listeners_mutex.push_back(mutex);
    finish_listeners_cond.push_back(cond);
}

void InferenceSession::notify_finish_listeners()
{
    for (int i = 0; i < finish_listeners_mutex.size(); i++)
    {
        pthread_mutex_lock(finish_listeners_mutex[i]);
        PRINT_THREAD_SUB("Notifying finish listener: " << instance_name);
        pthread_cond_signal(finish_listeners_cond[i]);
        pthread_mutex_unlock(finish_listeners_mutex[i]);
    }
}

void InferenceSession::reset_state()
{
    num_inferenced++;
    finish_time_ts = get_current_time_milliseconds();
    state = SESSION_STATE_IDLE;
}

void test_single_session(
    const std::string& model_filepath, const std::string& label_filepath, const std::string& image_filepath,
    int num_intra_threads, int num_inter_threads,
    int batch_size, int num_tests
) {
    printf(PRT_COLOR_CYAN "Single Session Inference\n" PRT_COLOR_RESET);

    std::string instance_name{"test-single-session"};

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
        printf("Elapsed time: %lld ms\n", elapsed);

    }
    session.print_results();

    float elapsed_single_avg = (float)std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0) / num_tests;
    std::cout << "Average elapsed time: " << elapsed_single_avg << " ms" << std::endl;
    std::cout << std::endl;
}

void test_multi_session(
    const std::string& model_filepath, const std::string& label_filepath, const std::string& image_filepath,
    int num_intra_threads, int num_inter_threads, int num_multi_threads,
    int batch_size, int num_tests
) {
    printf(PRT_COLOR_CYAN "Multi Session Inference\n" PRT_COLOR_RESET);
    printf("<Inference Information>\n");
    printf(" - Number of sessions: %d\n", num_multi_threads);
    printf("\n");

    std::string instance_name{"test-multi-session"};

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

        printf("Elapsed time: %lld ms\n", elapsed_total.back());
    }
    sessions[0]->print_results();

    float elapsed_multi_avg = (float)std::accumulate(elapsed_total.begin(), elapsed_total.end(), 0) / num_tests;
    std::cout << "Average elapsed time: " << elapsed_multi_avg << " ms" << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < num_multi_threads; i++) {
        delete sessions[i];
    }
}