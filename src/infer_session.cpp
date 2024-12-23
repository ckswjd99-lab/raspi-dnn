#include "infer_session.hpp"
#include "utils.hpp"
#include "input.hpp"

#define PREFIX_THREAD_SUB "\033[1;33m[ST]\033[0m "
#define PREFIX_THREAD_MAIN "\033[1;32m[MT]\033[0m "
#ifdef DEBUG_THREAD
#define PRINT_THREAD_SUB(msg) std::cout << PREFIX_THREAD_SUB << msg << std::endl
#define PRINT_THREAD_MAIN(msg) std::cout << PREFIX_THREAD_MAIN << msg << std::endl
#else
#define PRINT_THREAD_SUB(msg)
#define PRINT_THREAD_MAIN(msg)
#endif

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
}

InferenceSession::~InferenceSession() {
    delete session;
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

void InferenceSession::infer_sync()
{
    session->Run(
        Ort::RunOptions{nullptr}, 
        input_names.data(), input_tensors.data(), 1, 
        output_names.data(), output_tensors.data(), 1
    );
}

void InferenceSession::infer_async()
{
    if (flag_infer == 1) {
        PRINT_THREAD_MAIN("infer_async() called while infer_async() is already running.");
        return;
    }

    flag_infer = 1;
    pthread_create(&thread, nullptr, (void* (*)(void*))&InferenceSession::infer_sync, this);
}

void InferenceSession::wait_infer()
{
    if (flag_infer == 0) {
        PRINT_THREAD_MAIN("wait_infer() called while infer_async() is not running.");
        return;
    }

    pthread_join(thread, nullptr);
    flag_infer = 0;
}
