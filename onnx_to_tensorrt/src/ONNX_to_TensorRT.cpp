#include <iostream>
#include <fstream>
#include <memory>
#include "NvInfer.h"
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <set>

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <ONNX model path> <engine output path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " /path/to/model.onnx /path/to/model.engine" << std::endl;
        return 1;
    }

    std::string onnx_filename = argv[1];
    std::string engine_file = argv[2];

    std::cout << "ONNX file path: " << onnx_filename << std::endl;
    std::cout << "Engine output path: " << engine_file << std::endl;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builder->platformHasFastFp16()) {
        std::cerr << "[ERROR] Platform does not support fast FP16! Please check if JetPack/driver is properly installed." << std::endl;
        return -1;
    }

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 2ULL << 30);

    config->setFlag(nvinfer1::BuilderFlag::kFP16);  // Enable FP16 precision (float32 fallback)
    // config->clearFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    config->setTacticSources(1U << static_cast<int>(TacticSource::kCUBLAS));  // Use only cuBLAS for tactic selection
    // config->setMinTimingIterations(1);
    // config->setAvgTimingIterations(1);  // Reduce tactic search complexity


    uint32_t explicitBatch = 1U << static_cast<uint64_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));


    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    if (!parser->parseFromFile(onnx_filename.c_str(), static_cast<int64_t>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse the ONNX model." << std::endl;
        return -1;
    }
  
    std::vector<std::string> lockedFp32Layers;
    for (int i = 0; i < network->getNbLayers(); ++i) {
        auto* layer = network->getLayer(i);
        const char* layerNameCstr = layer->getName();
        std::string layerName = layerNameCstr ? layerNameCstr : "(unnamed)";
        std::cout << "layer name: " << layerName << std::endl;

        if (layerName.find("Div") != std::string::npos || 
            layerName.find("Pow") != std::string::npos || 
            layerName.find("Reduce") != std::string::npos ||
            layerName.find("Sqrt") != std::string::npos) {
            layer->setPrecision(DataType::kFLOAT);
            for (int out = 0; out < layer->getNbOutputs(); ++out) {
                // layer->setOutputType(out, DataType::kFLOAT);
            }
            lockedFp32Layers.push_back(layerName);
            continue;
        }
    }

    if (lockedFp32Layers.empty()) {
        std::cout << "No layers were marked as FP32. All layers are FP16 by default." << std::endl;
    } else {
        std::cout << "Locked FP32 layers: " <<  lockedFp32Layers.size() << std::endl;
        for (const auto& layer : lockedFp32Layers) {
            std::cout << layer << std::endl;
        }
    }    


    auto profile = builder->createOptimizationProfile();

  
    for (int i = 0; i < network->getNbInputs(); ++i) {
        const char* input_name = network->getInput(i)->getName();
        std::string name(input_name);
        std::cout << "Setting dimensions for input: " << input_name << std::endl;
        if(name == "inputs_embeds") {
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 2048));
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 64,2048));
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 512,2048));
        } else if (name == "position_ids") {
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 64));
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, 512));
        } else if (name == "attention_mask") {
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 64));
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, 512));
        } else if (name == "past_key_values") {
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(4, 32, 32));
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(4, 32, 32));
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(4, 32, 32));
        } else {
            std::cerr << "Unknown input name: " << input_name << std::endl;

        }

    }
    
    config->addOptimizationProfile(profile);
    config->setCalibrationProfile(profile);


    std::unique_ptr<IHostMemory> engine_stream{builder->buildSerializedNetwork(*network, *config)};
    if (!engine_stream) {
        std::cerr << "Failed to build serialized engine." << std::endl;
        return -1;
    }

    std::ofstream engine_out(engine_file, std::ios::binary);
    if (!engine_out) {
        std::cerr << "Failed to open file for writing engine: " << engine_file << std::endl;
        return -1;
    }
    engine_out.write(static_cast<const char*>(engine_stream->data()), engine_stream->size());
    engine_out.close();
    std::cout << "Engine saved to " << engine_file << std::endl;

    std::cout << "TensorRT Engine successfully created!" << std::endl;

    return 0;
}