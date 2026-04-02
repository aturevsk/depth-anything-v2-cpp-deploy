// Test harness for Depth-Anything-V2-Small NXP S32-compatible implementation
// No Accelerate framework, no STL containers, no std::ifstream
//
// Build:
//   clang++ -std=c++14 -O3 -ffp-contract=off -o depth_test_s32 main_s32.cpp
//
// Run:
//   ./depth_test_s32 ../weights ../reference

#include "depth_anything_v2_s32.h"
#include <cstdio>
#include <cmath>
#include <chrono>

// Static I/O buffers (no std::vector)
static float s_input[3 * 518 * 784];           // 4.65 MB
static float s_output[518 * 784];              // 1.55 MB
static float s_ref_output[518 * 784];          // 1.55 MB

static int load_binary(const char* path, float* dst, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }
    size_t read = fread(dst, sizeof(float), (size_t)n, f);
    fclose(f);
    return (int)read == n ? 0 : -1;
}

int main(int argc, char** argv) {
    const char* weights_dir = "../../weights";
    const char* ref_dir = "../../reference";

    if (argc > 1) weights_dir = argv[1];
    if (argc > 2) ref_dir = argv[2];

    printf("=== Depth-Anything-V2-Small: NXP S32-Compatible Implementation ===\n");
    printf("Properties: No BLAS, static allocation, no STL containers\n");
    printf("Target: NXP S32G / Renesas R-Car / TI TDA4VM (Cortex-A clusters)\n\n");

    // Load model weights
    printf("Loading weights from: %s\n", weights_dir);
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    dav2_s32::load_weights(weights_dir);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Weight load time: %.1f ms\n\n", load_ms);

    // Load reference input
    char path[512];
    snprintf(path, sizeof(path), "%s/input.bin", ref_dir);
    printf("Loading reference input from: %s\n", path);
    if (load_binary(path, s_input, 3 * 518 * 784) != 0) return 1;

    // Load reference output
    snprintf(path, sizeof(path), "%s/output.bin", ref_dir);
    printf("Loading reference output from: %s\n", path);
    if (load_binary(path, s_ref_output, 518 * 784) != 0) return 1;

    // Run inference
    printf("\nRunning inference...\n");
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    dav2_s32::forward(s_input, s_output);
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("Inference time: %.1f ms\n\n", infer_ms);

    // Compare with reference (explicit loops, no std::min_element/max_element)
    printf("=== Accuracy Comparison vs PyTorch Reference ===\n");
    double max_err = 0.0;
    double sum_err = 0.0;
    double sum_sq_err = 0.0;
    int n = 518 * 784;

    float ref_min = s_ref_output[0];
    float ref_max = s_ref_output[0];
    float out_min = s_output[0];
    float out_max = s_output[0];

    for (int i = 0; i < n; i++) {
        double err = std::abs((double)s_output[i] - (double)s_ref_output[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
        sum_sq_err += err * err;
        if (s_ref_output[i] < ref_min) ref_min = s_ref_output[i];
        if (s_ref_output[i] > ref_max) ref_max = s_ref_output[i];
        if (s_output[i] < out_min) out_min = s_output[i];
        if (s_output[i] > out_max) out_max = s_output[i];
    }

    double mean_err = sum_err / n;
    double rmse = std::sqrt(sum_sq_err / n);
    double ref_range = (double)ref_max - (double)ref_min;

    printf("Reference output range: [%.4f, %.4f]\n", ref_min, ref_max);
    printf("Our output range:       [%.4f, %.4f]\n", out_min, out_max);
    printf("Max absolute error:     %.6e\n", max_err);
    printf("Mean absolute error:    %.6e\n", mean_err);
    printf("RMSE:                   %.6e\n", rmse);
    printf("Relative RMSE:          %.6e\n", rmse / ref_range);

    // Benchmark (3 runs)
    printf("\n=== Benchmark (3 runs) ===\n");
    double total_ms = 0;
    for (int run = 0; run < 3; run++) {
        std::chrono::high_resolution_clock::time_point ta = std::chrono::high_resolution_clock::now();
        dav2_s32::forward(s_input, s_output);
        std::chrono::high_resolution_clock::time_point tb = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(tb - ta).count();
        printf("Run %d: %.1f ms\n", run + 1, ms);
        total_ms += ms;
    }
    printf("Average: %.1f ms\n", total_ms / 3.0);

    // Save output
    {
        FILE* f = fopen("s32_output.bin", "wb");
        if (f) {
            fwrite(s_output, sizeof(float), (size_t)n, f);
            fclose(f);
            printf("\nOutput saved to s32_output.bin\n");
        }
    }

    // Save results JSON
    {
        FILE* f = fopen("s32_results.json", "w");
        if (f) {
            fprintf(f, "{\n");
            fprintf(f, "  \"approach\": \"Manual C++ (S32-compatible)\",\n");
            fprintf(f, "  \"target\": \"NXP S32G / Cortex-A domain controllers\",\n");
            fprintf(f, "  \"blas\": false,\n");
            fprintf(f, "  \"static_allocation\": true,\n");
            fprintf(f, "  \"stl_containers\": false,\n");
            fprintf(f, "  \"load_time_ms\": %.1f,\n", load_ms);
            fprintf(f, "  \"inference_time_ms\": %.1f,\n", infer_ms);
            fprintf(f, "  \"avg_inference_ms\": %.1f,\n", total_ms / 3.0);
            fprintf(f, "  \"max_abs_error\": %.6e,\n", max_err);
            fprintf(f, "  \"mean_abs_error\": %.6e,\n", mean_err);
            fprintf(f, "  \"rmse\": %.6e,\n", rmse);
            fprintf(f, "  \"relative_rmse\": %.6e\n", rmse / ref_range);
            fprintf(f, "}\n");
            fclose(f);
            printf("Results saved to s32_results.json\n");
        }
    }

    return 0;
}
