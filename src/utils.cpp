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

std::vector<std::string> readLabels(const std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}