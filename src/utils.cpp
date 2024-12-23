#include "utils.hpp"

std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

std::vector<std::string> read_labels(const std::string& label_filepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(label_filepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

void print_inference_results(const std::vector<float>& output_tensor_values, const std::vector<std::string>& labels, int batch_size)
{
    std::vector<int> pred_ids(batch_size, 0);
    std::vector<std::string> pred_labels(batch_size);
    std::vector<float> confidences(batch_size, 0.0f);
    for (int64_t b = 0; b < batch_size; ++b)
    {
        float activation = 0;
        float max_activation = std::numeric_limits<float>::lowest();
        float exp_sum = 0;
        for (int i = 0; i < labels.size(); i++)
        {
            activation = output_tensor_values.at(i + b * labels.size());
            exp_sum += std::exp(activation);
            if (activation > max_activation)
            {
                pred_ids.at(b) = i;
                max_activation = activation;
            }
        }
        pred_labels.at(b) = labels.at(pred_ids.at(b));
        confidences.at(b) = std::exp(max_activation) / exp_sum;
    }
    for (int64_t b = 0; b < batch_size; ++b)
    {
        assert(("Output predictions should all be identical.", pred_ids.at(b) == pred_ids.at(0)));
    }

    std::cout << "Predicted Label ID: " << pred_ids.at(0) << std::endl;
    std::cout << "Predicted Label: " << pred_labels.at(0) << std::endl;
    std::cout << "Uncalibrated Confidence: " << confidences.at(0) << std::endl;
}