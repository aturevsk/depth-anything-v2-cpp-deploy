// Test harness for Depth-Anything-V2-Small manual C++ implementation
// Loads weights, runs inference, compares against PyTorch reference, benchmarks
#include "depth_anything_v2.h"
#include <cstdio>
#include <cmath>

int main(int argc, char** argv) {
    std::string weights_dir = "../../weights";
    std::string ref_dir = "../../reference";

    if (argc > 1) weights_dir = argv[1];
    if (argc > 2) ref_dir = argv[2];

    printf("=== Depth-Anything-V2-Small: Manual C++ Implementation ===\n\n");

    // Load model
    printf("Loading weights from: %s\n", weights_dir.c_str());
    dav2::DepthAnythingV2 model;
    auto t0 = std::chrono::high_resolution_clock::now();
    model.load_weights(weights_dir);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Weight load time: %.1f ms\n\n", load_ms);

    // Load reference input
    printf("Loading reference input from: %s/input.bin\n", ref_dir.c_str());
    std::vector<float> input(1 * 3 * 518 * 784);
    {
        std::ifstream f(ref_dir + "/input.bin", std::ios::binary);
        if (!f.is_open()) { fprintf(stderr, "Cannot open input.bin\n"); return 1; }
        f.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
    }

    // Load reference output
    printf("Loading reference output from: %s/output.bin\n", ref_dir.c_str());
    std::vector<float> ref_output(1 * 518 * 784);
    {
        std::ifstream f(ref_dir + "/output.bin", std::ios::binary);
        if (!f.is_open()) { fprintf(stderr, "Cannot open output.bin\n"); return 1; }
        f.read(reinterpret_cast<char*>(ref_output.data()), ref_output.size() * sizeof(float));
    }

    // Run inference
    printf("\nRunning inference...\n");
    std::vector<float> output(1 * 518 * 784);

    auto t2 = std::chrono::high_resolution_clock::now();
    model.forward(input.data(), output.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("Inference time: %.1f ms\n\n", infer_ms);

    // Compare with reference
    printf("=== Accuracy Comparison vs PyTorch Reference ===\n");
    double max_err = 0.0;
    double sum_err = 0.0;
    double sum_sq_err = 0.0;
    double sum_ref = 0.0;
    double sum_ref_sq = 0.0;
    int n = 518 * 784;

    for (int i = 0; i < n; i++) {
        double err = std::abs((double)output[i] - (double)ref_output[i]);
        max_err = std::max(max_err, err);
        sum_err += err;
        sum_sq_err += err * err;
        sum_ref += ref_output[i];
        sum_ref_sq += (double)ref_output[i] * ref_output[i];
    }

    double mean_err = sum_err / n;
    double rmse = std::sqrt(sum_sq_err / n);
    double ref_mean = sum_ref / n;
    double ref_range = 0;
    {
        float mn = *std::min_element(ref_output.begin(), ref_output.end());
        float mx = *std::max_element(ref_output.begin(), ref_output.end());
        ref_range = mx - mn;
        printf("Reference output range: [%.4f, %.4f]\n", mn, mx);
    }
    {
        float mn = *std::min_element(output.begin(), output.end());
        float mx = *std::max_element(output.begin(), output.end());
        printf("Our output range:       [%.4f, %.4f]\n", mn, mx);
    }
    printf("Max absolute error:     %.6e\n", max_err);
    printf("Mean absolute error:    %.6e\n", mean_err);
    printf("RMSE:                   %.6e\n", rmse);
    printf("Relative RMSE:          %.6e\n", rmse / ref_range);

    // Benchmark (3 runs)
    printf("\n=== Benchmark (3 runs) ===\n");
    double total_ms = 0;
    for (int run = 0; run < 3; run++) {
        auto ta = std::chrono::high_resolution_clock::now();
        model.forward(input.data(), output.data());
        auto tb = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(tb - ta).count();
        printf("Run %d: %.1f ms\n", run + 1, ms);
        total_ms += ms;
    }
    printf("Average: %.1f ms\n", total_ms / 3.0);

    // Save output
    {
        std::ofstream f("manual_cpp_output.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    }
    printf("\nOutput saved to manual_cpp_output.bin\n");

    // Save results JSON
    {
        FILE* f = fopen("manual_cpp_results.json", "w");
        fprintf(f, "{\n");
        fprintf(f, "  \"approach\": \"Manual C++\",\n");
        fprintf(f, "  \"load_time_ms\": %.1f,\n", load_ms);
        fprintf(f, "  \"inference_time_ms\": %.1f,\n", infer_ms);
        fprintf(f, "  \"avg_inference_ms\": %.1f,\n", total_ms / 3.0);
        fprintf(f, "  \"max_abs_error\": %.6e,\n", max_err);
        fprintf(f, "  \"mean_abs_error\": %.6e,\n", mean_err);
        fprintf(f, "  \"rmse\": %.6e,\n", rmse);
        fprintf(f, "  \"relative_rmse\": %.6e\n", rmse / ref_range);
        fprintf(f, "}\n");
        fclose(f);
    }
    printf("Results saved to manual_cpp_results.json\n");

    return 0;
}
