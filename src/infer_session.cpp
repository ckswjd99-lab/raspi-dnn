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
static T vectorProduct(const std::vector<T>& v)
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

Ort::Session *createSession(const std::string& modelFilepath, const std::string& instanceName, int numIntraThreads, int numInterThreads)
{
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(numIntraThreads);
    sessionOptions.SetInterOpNumThreads(numInterThreads);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    return new Ort::Session(env, modelFilepath.c_str(), sessionOptions);
}

InferenceSession::InferenceSession(
    std::string instanceName,
    const std::string& model_path, const std::string& label_path,
    int num_intra_threads, int num_inter_threads
) : instanceName(instanceName), model_path(model_path), label_path(label_path), num_intra_threads(num_intra_threads), num_inter_threads(num_inter_threads)
{
    labels = readLabels(label_path);
    session = createSession(model_path, instanceName, num_intra_threads, num_inter_threads);

    pthread_mutex_init(&mutex, nullptr);
    pthread_cond_init(&cond, nullptr);
}

InferenceSession::~InferenceSession() {
    delete session;
}

void InferenceSession::print_info() {
    printf("<Model Information>\n");
    printf("Instance Name: %s\n", instanceName.c_str());
    printf("Model Path: %s\n", model_path.c_str());
    printf("Label Path: %s\n", label_path.c_str());
    printf("Number of (Intra, Inter) Threads: (%d, %d)\n", num_intra_threads, num_inter_threads);
    printf("\n");
}

void InferenceSession::load_input(const std::string& image_path, int batch_size)
{
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session->GetInputCount();
    size_t numOutputNodes = session->GetOutputCount();

    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting input batch size to " << batch_size << "." << std::endl;
        inputDims.at(0) = batch_size;
    }

    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting output batch size to " << batch_size << "." << std::endl;
        outputDims.at(0) = batch_size;
    }

    size_t inputTensorSize = vectorProduct(inputDims);
    inputTensorValues.resize(inputTensorSize);
    prepareInputTensor(image_path, inputDims, inputTensorValues, batch_size, inputTensorSize);

    size_t outputTensorSize = vectorProduct(outputDims);
    assert(("Output tensor size should equal to the label set size.", labels.size() * batch_size == outputTensorSize));
    outputTensorValues.resize(outputTensorSize);

    auto inputNodesNum = session->GetInputCount();
    for (int i = 0; i < inputNodesNum; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        inputNames.push_back(inputNodeNameAllocatedStrings.back().get());
    }

    auto outputNodesNum = session->GetOutputCount();
    for (int i = 0; i < outputNodesNum; i++) {
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize, outputDims.data(), outputDims.size()));

}

void InferenceSession::print_results()
{
    outputTensorValues.assign(outputTensors.at(0).GetTensorMutableData<float>(), outputTensors.at(0).GetTensorMutableData<float>() + outputTensors.at(0).GetTensorTypeAndShapeInfo().GetElementCount());
    printInferenceResults(outputTensorValues, labels, 1);
}

void InferenceSession::run()
{
    pthread_mutex_lock(&mutex);
    run_flag = 1;
    pthread_mutex_unlock(&mutex);

    pthread_create(&thread, nullptr, (void* (*)(void*))&InferenceSession::thread_func, this);
}

void InferenceSession::wait_load()
{
    while (true) {
        pthread_mutex_lock(&mutex);
        int flag = load_flag;
        pthread_mutex_unlock(&mutex);

        if (flag == 1) break;
    }

    PRINT_THREAD_MAIN("Model loaded");
}

void InferenceSession::infer()
{
    PRINT_THREAD_MAIN("Signaling inference");

    pthread_mutex_lock(&mutex);
    infer_flag = 1;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
}

void InferenceSession::wait_infer()
{
    while (true) {
        pthread_mutex_lock(&mutex);
        int flag = infer_flag;
        pthread_mutex_unlock(&mutex);

        if (flag == 0) break;
    }

    PRINT_THREAD_MAIN("Inference done");
}

void InferenceSession::stop()
{
    PRINT_THREAD_MAIN("Signaling stop");

    pthread_mutex_lock(&mutex);
    run_flag = 0;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);

    pthread_join(thread, nullptr);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
}

void *InferenceSession::thread_func()
{
    PRINT_THREAD_SUB("Thread func started");
    load_flag = 1;

    while (true)
    {
        PRINT_THREAD_SUB("Thread func waiting");

        pthread_mutex_lock(&mutex);
        infer_flag = 0;
        pthread_cond_wait(&cond, &mutex);
        pthread_mutex_unlock(&mutex);
        
        PRINT_THREAD_SUB("Thread func signaled");

        if (run_flag == 0) break;

        printf("<Running inference>\n");

        auto begin = std::chrono::steady_clock::now();
        session->Run(
            Ort::RunOptions{nullptr}, 
            inputNames.data(), inputTensors.data(), 1, 
            outputNames.data(), outputTensors.data(), 1
        );
        auto end = std::chrono::steady_clock::now();

        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        
        std::cout << "Inference Time: " << elapsed_ms.count() << " ms" << std::endl;
        print_results();

    }

    return nullptr;
}

