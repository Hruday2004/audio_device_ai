#include <jni.h>
#include <string>
#include <vector>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include "nnapi_provider_factory.h"
#include <android/log.h>

#define LOG_TAG "ONNXRuntime"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jstring
JNICALL
Java_com_example_noise_1reduceer_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jfloatArray
JNICALL
Java_com_example_noise_1reduceer_MainActivity_runInference(JNIEnv* env, jobject /* this */, jfloatArray inputArray, jstring modelPath) {

    const char *model_path = env->GetStringUTFChars(modelPath, nullptr);
    std::unique_ptr<Ort::Env> environment(new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE,"test"));

    LOGD("model path  %s ", model_path);
    // Initialize ONNX Runtime
//    Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING, "test");
//    Ort::SessionOptions session_options;
//    Ort::Session session(e, model_path, session_options);

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    uint32_t nnapi_flags = 0;
    nnapi_flags |= NNAPI_FLAG_USE_FP16;
    OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flags);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    std::unique_ptr<Ort::Session> session  = std::make_unique<Ort::Session>(*environment.get(), model_path, session_options);

    LOGD("session create completed");
    // Convert Java float array to C++ float array
    jfloat* input_elements = env->GetFloatArrayElements(inputArray, NULL);
    jsize input_length = env->GetArrayLength(inputArray);

    // Define input tensor shape and values
    std::vector<int64_t> input_shape{1, 10};
    std::vector<float> input_values(input_elements, input_elements + input_length);

    // Release the Java float array
    env->ReleaseFloatArrayElements(inputArray, input_elements, 0);

    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size());

    // Prepare input and output node names
    const char* input_node_names[] = {"input"};
    const char* output_node_names[] = {"output"};

    // Run inference
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names, &input_tensor, 1, output_node_names, 1);

    // Get output
    float* output_values = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> output_vector(output_values, output_values + 2);

    // Convert C++ float array to Java float array
    jfloatArray outputArray = env->NewFloatArray(output_vector.size());
    env->SetFloatArrayRegion(outputArray, 0, output_vector.size(), output_vector.data());

    return outputArray;
}