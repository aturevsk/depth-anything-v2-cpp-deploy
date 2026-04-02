#pragma once
// Depth-Anything-V2-Small: Manual C++ Implementation for Embedded Deployment
// Architecture: DINOv2-Small ViT encoder + DPT decoder
// Input:  [1, 3, 518, 784] float32  (NCHW)
// Output: [1, 518, 784] float32     (depth map)

#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define USE_ACCELERATE 1
#else
#define USE_ACCELERATE 0
#endif

// ============================================================================
// Model Constants
// ============================================================================
namespace dav2 {

constexpr int BATCH = 1;
constexpr int IN_C = 3;
constexpr int IN_H = 518;
constexpr int IN_W = 784;
constexpr int PATCH = 14;
constexpr int GRID_H = IN_H / PATCH;  // 37
constexpr int GRID_W = IN_W / PATCH;  // 56
constexpr int NUM_PATCHES = GRID_H * GRID_W;  // 2072
constexpr int SEQ_LEN = NUM_PATCHES + 1;       // 2073 (with CLS)
constexpr int EMBED = 384;
constexpr int NUM_HEADS = 6;
constexpr int HEAD_DIM = EMBED / NUM_HEADS;  // 64
constexpr int MLP_DIM = 1536;  // 4 * EMBED
constexpr int NUM_BLOCKS = 12;
constexpr int POS_EMBED_ORIG = 1370;  // original pos embed length (for interpolation)

// Decoder constants
constexpr int DPT_FEATURES = 64;
constexpr int OUT_H = IN_H;  // 518
constexpr int OUT_W = IN_W;  // 784

// ============================================================================
// Utility: flat indexing helpers (NCHW layout)
// ============================================================================
inline int idx4(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}
inline int idx3(int c, int h, int w, int H, int W) {
    return (c * H + h) * W + w;
}
inline int idx2(int r, int c, int C) {
    return r * C + c;
}

// ============================================================================
// BLAS Wrappers
// ============================================================================
inline void matmul(const float* A, const float* B, float* C,
                   int M, int N, int K, bool transA = false, bool transB = false) {
#if USE_ACCELERATE
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K, 1.0f, A, transA ? M : K, B, transB ? K : N,
                0.0f, C, N);
#else
    // Naive fallback
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = transA ? A[k * M + m] : A[m * K + k];
                float b = transB ? B[n * K + k] : B[k * N + n];
                sum += a * b;
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

inline void matmul_add(const float* A, const float* B, float* C,
                       const float* bias, int M, int N, int K) {
    matmul(A, B, C, M, N, K, false, true);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C[m * N + n] += bias[n];
}

// ============================================================================
// Activation Functions
// ============================================================================
inline float gelu(float x) {
    // Use erf-based GELU (matches PyTorch aten.gelu.default)
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));  // 1/sqrt(2)
}

inline void gelu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = gelu(x[i]);
}

inline void relu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = std::max(0.0f, x[i]);
}

// ============================================================================
// Layer Norm
// ============================================================================
inline void layer_norm(const float* input, float* output, const float* gamma,
                       const float* beta, int seq_len, int dim, float eps = 1e-5f) {
    for (int s = 0; s < seq_len; s++) {
        const float* in_row = input + s * dim;
        float* out_row = output + s * dim;

        float mean = 0.0f;
        for (int d = 0; d < dim; d++) mean += in_row[d];
        mean /= dim;

        float var = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = in_row[d] - mean;
            var += diff * diff;
        }
        var /= dim;

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int d = 0; d < dim; d++) {
            out_row[d] = (in_row[d] - mean) * inv_std * gamma[d] + beta[d];
        }
    }
}

// ============================================================================
// Softmax
// ============================================================================
inline void softmax_inplace(float* x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* row = x + r * cols;
        float max_val = *std::max_element(row, row + cols);
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = std::exp(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) row[c] *= inv_sum;
    }
}

// ============================================================================
// Conv2d (direct implementation with optional padding)
// ============================================================================
void conv2d(const float* input, float* output,
            const float* weight, const float* bias,
            int in_c, int in_h, int in_w,
            int out_c, int kh, int kw,
            int stride_h, int stride_w,
            int pad_h, int pad_w) {
    int out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;

    for (int oc = 0; oc < out_c; oc++) {
        float b = bias ? bias[oc] : 0.0f;
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = b;
                for (int ic = 0; ic < in_c; ic++) {
                    for (int fh = 0; fh < kh; fh++) {
                        for (int fw = 0; fw < kw; fw++) {
                            int ih = oh * stride_h - pad_h + fh;
                            int iw = ow * stride_w - pad_w + fw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += input[idx3(ic, ih, iw, in_h, in_w)] *
                                       weight[((oc * in_c + ic) * kh + fh) * kw + fw];
                            }
                        }
                    }
                }
                output[idx3(oc, oh, ow, out_h, out_w)] = sum;
            }
        }
    }
}

// No-bias variant
void conv2d_nobias(const float* input, float* output,
                   const float* weight,
                   int in_c, int in_h, int in_w,
                   int out_c, int kh, int kw,
                   int stride_h, int stride_w,
                   int pad_h, int pad_w) {
    conv2d(input, output, weight, nullptr, in_c, in_h, in_w, out_c, kh, kw,
           stride_h, stride_w, pad_h, pad_w);
}

// ============================================================================
// ConvTranspose2d
// ============================================================================
void conv_transpose2d(const float* input, float* output,
                      const float* weight, const float* bias,
                      int in_c, int in_h, int in_w,
                      int out_c, int kh, int kw,
                      int stride_h, int stride_w,
                      int pad_h, int pad_w) {
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kh;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kw;

    // Initialize with bias
    for (int oc = 0; oc < out_c; oc++) {
        float b = bias ? bias[oc] : 0.0f;
        for (int oh = 0; oh < out_h; oh++)
            for (int ow = 0; ow < out_w; ow++)
                output[idx3(oc, oh, ow, out_h, out_w)] = b;
    }

    // Scatter-add
    for (int ic = 0; ic < in_c; ic++) {
        for (int ih = 0; ih < in_h; ih++) {
            for (int iw = 0; iw < in_w; iw++) {
                float val = input[idx3(ic, ih, iw, in_h, in_w)];
                for (int fh = 0; fh < kh; fh++) {
                    for (int fw = 0; fw < kw; fw++) {
                        int oh = ih * stride_h - pad_h + fh;
                        int ow = iw * stride_w - pad_w + fw;
                        if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                            for (int oc = 0; oc < out_c; oc++) {
                                // weight: [in_c, out_c, kh, kw]
                                output[idx3(oc, oh, ow, out_h, out_w)] +=
                                    val * weight[((ic * out_c + oc) * kh + fh) * kw + fw];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Bilinear Upsample 2D
// ============================================================================
void upsample_bilinear2d(const float* input, float* output,
                         int channels, int in_h, int in_w,
                         int out_h, int out_w) {
    // align_corners=True behavior (matches PyTorch default for this model)
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float ih_f = (out_h > 1) ? (float)oh * (in_h - 1) / (out_h - 1) : 0.0f;
                float iw_f = (out_w > 1) ? (float)ow * (in_w - 1) / (out_w - 1) : 0.0f;

                int ih0 = (int)ih_f;
                int iw0 = (int)iw_f;
                int ih1 = std::min(ih0 + 1, in_h - 1);
                int iw1 = std::min(iw0 + 1, in_w - 1);

                float h_frac = ih_f - ih0;
                float w_frac = iw_f - iw0;

                float v00 = input[idx3(c, ih0, iw0, in_h, in_w)];
                float v01 = input[idx3(c, ih0, iw1, in_h, in_w)];
                float v10 = input[idx3(c, ih1, iw0, in_h, in_w)];
                float v11 = input[idx3(c, ih1, iw1, in_h, in_w)];

                float val = v00 * (1 - h_frac) * (1 - w_frac) +
                            v01 * (1 - h_frac) * w_frac +
                            v10 * h_frac * (1 - w_frac) +
                            v11 * h_frac * w_frac;

                output[idx3(c, oh, ow, out_h, out_w)] = val;
            }
        }
    }
}

// ============================================================================
// Bicubic Upsample 2D
// ============================================================================
inline float cubic_interp(float x) {
    // PyTorch uses a=-0.75 coefficient (not Catmull-Rom a=-0.5)
    float ax = std::abs(x);
    if (ax <= 1.0f) return ((1.25f * ax - 2.25f) * ax) * ax + 1.0f;
    if (ax < 2.0f) return ((-0.75f * ax + 3.75f) * ax - 6.0f) * ax + 3.0f;
    return 0.0f;
}

void upsample_bicubic2d(const float* input, float* output,
                        int channels, int in_h, int in_w,
                        int out_h, int out_w,
                        bool align_corners = true) {
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float ih_f, iw_f;
                if (align_corners) {
                    ih_f = (out_h > 1) ? (float)oh * (in_h - 1) / (out_h - 1) : 0.0f;
                    iw_f = (out_w > 1) ? (float)ow * (in_w - 1) / (out_w - 1) : 0.0f;
                } else {
                    ih_f = ((float)oh + 0.5f) * in_h / out_h - 0.5f;
                    iw_f = ((float)ow + 0.5f) * in_w / out_w - 0.5f;
                }

                int ih_center = (int)std::floor(ih_f);
                int iw_center = (int)std::floor(iw_f);

                float val = 0.0f;
                for (int dh = -1; dh <= 2; dh++) {
                    for (int dw = -1; dw <= 2; dw++) {
                        int ih = std::clamp(ih_center + dh, 0, in_h - 1);
                        int iw = std::clamp(iw_center + dw, 0, in_w - 1);
                        float wh = cubic_interp(ih_f - (ih_center + dh));
                        float ww = cubic_interp(iw_f - (iw_center + dw));
                        val += input[idx3(c, ih, iw, in_h, in_w)] * wh * ww;
                    }
                }
                output[idx3(c, oh, ow, out_h, out_w)] = val;
            }
        }
    }
}

// ============================================================================
// Weight Storage
// ============================================================================
struct Weights {
    // Encoder
    std::vector<float> cls_token;         // [1, 1, 384]
    std::vector<float> pos_embed;         // [1, 1370, 384]
    std::vector<float> patch_proj_w;      // [384, 3, 14, 14]
    std::vector<float> patch_proj_b;      // [384]

    // Per-block
    struct Block {
        float ls1_gamma[EMBED];
        float ls2_gamma[EMBED];
        std::vector<float> norm1_w, norm1_b;         // [384]
        std::vector<float> norm2_w, norm2_b;         // [384]
        std::vector<float> attn_qkv_w, attn_qkv_b;  // [1152, 384], [1152]
        std::vector<float> attn_proj_w, attn_proj_b; // [384, 384], [384]
        std::vector<float> mlp_fc1_w, mlp_fc1_b;     // [1536, 384], [1536]
        std::vector<float> mlp_fc2_w, mlp_fc2_b;     // [384, 1536], [384]
    };
    Block blocks[NUM_BLOCKS];

    std::vector<float> norm_w, norm_b;  // final norm [384]

    // DPT Decoder
    struct {
        std::vector<float> proj0_w, proj0_b;  // [48, 384, 1, 1]
        std::vector<float> proj1_w, proj1_b;  // [96, 384, 1, 1]
        std::vector<float> proj2_w, proj2_b;  // [192, 384, 1, 1]
        std::vector<float> proj3_w, proj3_b;  // [384, 384, 1, 1]

        std::vector<float> resize0_w, resize0_b;  // ConvTranspose [48, 48, 4, 4]
        std::vector<float> resize1_w, resize1_b;  // ConvTranspose [96, 96, 2, 2]
        std::vector<float> resize3_w, resize3_b;  // Conv [384, 384, 3, 3] stride=2

        // layer_rn (1x1 conv, no bias)
        std::vector<float> layer1_rn_w;  // [64, 48, 3, 3] (scratch read says 3x3)
        std::vector<float> layer2_rn_w;  // [64, 96, 3, 3]
        std::vector<float> layer3_rn_w;  // [64, 192, 3, 3]
        std::vector<float> layer4_rn_w;  // [64, 384, 3, 3]

        // refinenet stages (each has resconfunit1 (optional) + resconfunit2 + out_conv)
        struct RefineStage {
            bool has_rcu1;
            std::vector<float> rcu1_conv1_w, rcu1_conv1_b;
            std::vector<float> rcu1_conv2_w, rcu1_conv2_b;
            std::vector<float> rcu2_conv1_w, rcu2_conv1_b;
            std::vector<float> rcu2_conv2_w, rcu2_conv2_b;
            std::vector<float> out_conv_w, out_conv_b;  // 1x1
        };
        RefineStage refine[4];  // stages 1-4

        // Output head
        std::vector<float> out_conv1_w, out_conv1_b;    // [32, 64, 3, 3]
        std::vector<float> out_conv2_0_w, out_conv2_0_b; // [32, 32, 3, 3]
        std::vector<float> out_conv2_2_w, out_conv2_2_b; // [1, 32, 1, 1]
    } dpt;

    // Load a single weight file
    static std::vector<float> load_bin(const std::string& path, int expected_numel) {
        std::vector<float> data(expected_numel);
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            fprintf(stderr, "ERROR: Cannot open weight file: %s\n", path.c_str());
            std::fill(data.begin(), data.end(), 0.0f);
            return data;
        }
        f.read(reinterpret_cast<char*>(data.data()), expected_numel * sizeof(float));
        return data;
    }

    void load(const std::string& dir) {
        auto L = [&](const std::string& name, int n) {
            return load_bin(dir + "/" + name + ".bin", n);
        };

        cls_token = L("pretrained_cls_token", 1 * 1 * EMBED);
        pos_embed = L("pretrained_pos_embed", 1 * POS_EMBED_ORIG * EMBED);
        patch_proj_w = L("pretrained_patch_embed_proj_weight", EMBED * IN_C * PATCH * PATCH);
        patch_proj_b = L("pretrained_patch_embed_proj_bias", EMBED);
        norm_w = L("pretrained_norm_weight", EMBED);
        norm_b = L("pretrained_norm_bias", EMBED);

        for (int i = 0; i < NUM_BLOCKS; i++) {
            auto& b = blocks[i];
            std::string prefix = "pretrained_blocks_" + std::to_string(i) + "_";
            auto ls1 = L(prefix + "ls1_gamma", EMBED);
            auto ls2 = L(prefix + "ls2_gamma", EMBED);
            std::memcpy(b.ls1_gamma, ls1.data(), EMBED * sizeof(float));
            std::memcpy(b.ls2_gamma, ls2.data(), EMBED * sizeof(float));

            b.norm1_w = L(prefix + "norm1_weight", EMBED);
            b.norm1_b = L(prefix + "norm1_bias", EMBED);
            b.norm2_w = L(prefix + "norm2_weight", EMBED);
            b.norm2_b = L(prefix + "norm2_bias", EMBED);
            b.attn_qkv_w = L(prefix + "attn_qkv_weight", 3 * EMBED * EMBED);
            b.attn_qkv_b = L(prefix + "attn_qkv_bias", 3 * EMBED);
            b.attn_proj_w = L(prefix + "attn_proj_weight", EMBED * EMBED);
            b.attn_proj_b = L(prefix + "attn_proj_bias", EMBED);
            b.mlp_fc1_w = L(prefix + "mlp_fc1_weight", MLP_DIM * EMBED);
            b.mlp_fc1_b = L(prefix + "mlp_fc1_bias", MLP_DIM);
            b.mlp_fc2_w = L(prefix + "mlp_fc2_weight", EMBED * MLP_DIM);
            b.mlp_fc2_b = L(prefix + "mlp_fc2_bias", EMBED);
        }

        // DPT head
        dpt.proj0_w = L("depth_head_projects_0_weight", 48 * EMBED * 1 * 1);
        dpt.proj0_b = L("depth_head_projects_0_bias", 48);
        dpt.proj1_w = L("depth_head_projects_1_weight", 96 * EMBED * 1 * 1);
        dpt.proj1_b = L("depth_head_projects_1_bias", 96);
        dpt.proj2_w = L("depth_head_projects_2_weight", 192 * EMBED * 1 * 1);
        dpt.proj2_b = L("depth_head_projects_2_bias", 192);
        dpt.proj3_w = L("depth_head_projects_3_weight", EMBED * EMBED * 1 * 1);
        dpt.proj3_b = L("depth_head_projects_3_bias", EMBED);

        dpt.resize0_w = L("depth_head_resize_layers_0_weight", 48 * 48 * 4 * 4);
        dpt.resize0_b = L("depth_head_resize_layers_0_bias", 48);
        dpt.resize1_w = L("depth_head_resize_layers_1_weight", 96 * 96 * 2 * 2);
        dpt.resize1_b = L("depth_head_resize_layers_1_bias", 96);
        dpt.resize3_w = L("depth_head_resize_layers_3_weight", EMBED * EMBED * 3 * 3);
        dpt.resize3_b = L("depth_head_resize_layers_3_bias", EMBED);

        dpt.layer1_rn_w = L("depth_head_scratch_layer1_rn_weight", 64 * 48 * 3 * 3);
        dpt.layer2_rn_w = L("depth_head_scratch_layer2_rn_weight", 64 * 96 * 3 * 3);
        dpt.layer3_rn_w = L("depth_head_scratch_layer3_rn_weight", 64 * 192 * 3 * 3);
        dpt.layer4_rn_w = L("depth_head_scratch_layer4_rn_weight", 64 * EMBED * 3 * 3);

        // RefineNet stages
        // Stage 4: only rcu2
        dpt.refine[3].has_rcu1 = false;
        dpt.refine[3].rcu2_conv1_w = L("depth_head_scratch_refinenet4_resconfunit2_conv1_weight", 64*64*3*3);
        dpt.refine[3].rcu2_conv1_b = L("depth_head_scratch_refinenet4_resconfunit2_conv1_bias", 64);
        dpt.refine[3].rcu2_conv2_w = L("depth_head_scratch_refinenet4_resconfunit2_conv2_weight", 64*64*3*3);
        dpt.refine[3].rcu2_conv2_b = L("depth_head_scratch_refinenet4_resconfunit2_conv2_bias", 64);
        dpt.refine[3].out_conv_w = L("depth_head_scratch_refinenet4_out_conv_weight", 64*64*1*1);
        dpt.refine[3].out_conv_b = L("depth_head_scratch_refinenet4_out_conv_bias", 64);

        // Stages 3, 2, 1: have both rcu1 and rcu2
        for (int s = 0; s < 3; s++) {
            int stage = 3 - s;  // 3, 2, 1
            std::string prefix = "depth_head_scratch_refinenet" + std::to_string(stage) + "_";
            dpt.refine[stage-1].has_rcu1 = true;
            dpt.refine[stage-1].rcu1_conv1_w = L(prefix + "resconfunit1_conv1_weight", 64*64*3*3);
            dpt.refine[stage-1].rcu1_conv1_b = L(prefix + "resconfunit1_conv1_bias", 64);
            dpt.refine[stage-1].rcu1_conv2_w = L(prefix + "resconfunit1_conv2_weight", 64*64*3*3);
            dpt.refine[stage-1].rcu1_conv2_b = L(prefix + "resconfunit1_conv2_bias", 64);
            dpt.refine[stage-1].rcu2_conv1_w = L(prefix + "resconfunit2_conv1_weight", 64*64*3*3);
            dpt.refine[stage-1].rcu2_conv1_b = L(prefix + "resconfunit2_conv1_bias", 64);
            dpt.refine[stage-1].rcu2_conv2_w = L(prefix + "resconfunit2_conv2_weight", 64*64*3*3);
            dpt.refine[stage-1].rcu2_conv2_b = L(prefix + "resconfunit2_conv2_bias", 64);
            dpt.refine[stage-1].out_conv_w = L(prefix + "out_conv_weight", 64*64*1*1);
            dpt.refine[stage-1].out_conv_b = L(prefix + "out_conv_bias", 64);
        }

        dpt.out_conv1_w = L("depth_head_scratch_output_conv1_weight", 32*64*3*3);
        dpt.out_conv1_b = L("depth_head_scratch_output_conv1_bias", 32);
        dpt.out_conv2_0_w = L("depth_head_scratch_output_conv2_0_weight", 32*32*3*3);
        dpt.out_conv2_0_b = L("depth_head_scratch_output_conv2_0_bias", 32);
        dpt.out_conv2_2_w = L("depth_head_scratch_output_conv2_2_weight", 1*32*1*1);
        dpt.out_conv2_2_b = L("depth_head_scratch_output_conv2_2_bias", 1);
    }
};

// ============================================================================
// Model: Depth-Anything-V2-Small
// ============================================================================
class DepthAnythingV2 {
public:
    Weights w;

    void load_weights(const std::string& weights_dir) {
        w.load(weights_dir);
    }

    // Main inference: input [1, 3, 518, 784], output [1, 518, 784]
    void forward(const float* input, float* output) {
        // === ENCODER: DINOv2-Small ViT ===

        // 1. Patch embedding: Conv2d(3, 384, 14, stride=14)
        std::vector<float> patch_embed(EMBED * GRID_H * GRID_W);
        conv2d(input, patch_embed.data(), w.patch_proj_w.data(), w.patch_proj_b.data(),
               IN_C, IN_H, IN_W, EMBED, PATCH, PATCH, PATCH, PATCH, 0, 0);

        // 2. Reshape to sequence: [384, 37, 56] -> [2072, 384]
        // Then prepend CLS token -> [2073, 384]
        std::vector<float> tokens(SEQ_LEN * EMBED);
        // CLS token
        std::memcpy(tokens.data(), w.cls_token.data(), EMBED * sizeof(float));
        // Flatten patches: from [C, H, W] to [H*W, C]
        for (int h = 0; h < GRID_H; h++) {
            for (int ww = 0; ww < GRID_W; ww++) {
                int patch_idx = h * GRID_W + ww;
                for (int c = 0; c < EMBED; c++) {
                    tokens[(patch_idx + 1) * EMBED + c] =
                        patch_embed[c * GRID_H * GRID_W + h * GRID_W + ww];
                }
            }
        }

        // 3. Add positional embeddings (interpolated from 1370 to 2073)
        add_pos_embed(tokens.data());

        // 4. Transformer blocks - capture intermediate outputs
        // Graph analysis shows features extracted at add_6, add_12, add_18, add_24
        // add_6 = block 3 attention residual (after attn, before MLP)
        // add_12 = block 6 attention residual
        // add_18 = block 9 attention residual
        // add_24 = block 11 full output (after MLP)
        // Actually: each block creates 2 adds. add index = block*2 + {0=attn, 1=mlp}
        // add_6 = block 3 attn add, add_12 = block 6 attn add, add_18 = block 9 attn add
        // add_24 = block 12... but there are only 12 blocks (0-11)
        // Wait: 24/2 = 12, add_24 = block 12 attn... can't be right.
        // Let me recount: add_0=b0_attn, add_1=b0_mlp, ..., add_22=b11_attn, add_23=b11_mlp
        // add_24 is at node 577 which is the 25th encoder add.
        // Looking at the output: add_24 input is add_23 (block 11 mlp) + mul_35 (ls2)
        // So add_24 IS the block 11 MLP residual output = full block 11 output
        // But add_6 (index 6): add_0,1=b0, add_2,3=b1, add_4,5=b2, add_6=b3_attn
        // So: add_6=b3 mid, add_12=b6 mid, add_18=b9 mid, add_24=b12_doesn't_exist
        // Actually add_24 node 577 feeds layer_norm_27. Let me reconsider.
        // The adds include the pos_embed addition: add at node 253 is pos_embed+token
        // So there are 25 adds: 1 (pos embed) + 24 (12 blocks * 2)
        // add_0 = pos embed add
        // add_1 = block 0 attn, add_2 = block 0 mlp
        // add_3 = block 1 attn, add_4 = block 1 mlp
        // add_5 = block 2 attn, add_6 = block 2 mlp  <-- feature 0
        // add_7 = block 3 attn, add_8 = block 3 mlp
        // add_9 = block 4 attn, add_10 = block 4 mlp
        // add_11 = block 5 attn, add_12 = block 5 mlp  <-- feature 1
        // add_13 = block 6 attn, add_14 = block 6 mlp
        // add_15 = block 7 attn, add_16 = block 7 mlp
        // add_17 = block 8 attn, add_18 = block 8 mlp  <-- feature 2
        // add_19 = block 9 attn, add_20 = block 9 mlp
        // add_21 = block 10 attn, add_22 = block 10 mlp
        // add_23 = block 11 attn, add_24 = block 11 mlp  <-- feature 3
        // Features extracted after blocks 2, 5, 8, 11 (complete block outputs)
        std::vector<float> tmp(SEQ_LEN * EMBED);
        std::vector<float> tmp2(SEQ_LEN * EMBED);

        constexpr int FEATURE_BLOCKS[4] = {2, 5, 8, 11};
        std::vector<float> block_outputs[4];
        int feat_idx = 0;

        for (int i = 0; i < NUM_BLOCKS; i++) {
            transformer_block(tokens.data(), tmp.data(), tmp2.data(), w.blocks[i]);

            if (feat_idx < 4 && i == FEATURE_BLOCKS[feat_idx]) {
                block_outputs[feat_idx].resize(SEQ_LEN * EMBED);
                std::memcpy(block_outputs[feat_idx].data(), tokens.data(), SEQ_LEN * EMBED * sizeof(float));
                feat_idx++;
            }
        }

        // === DECODER: DPT Head ===
        // Apply final norm to each intermediate output, extract patch tokens (skip CLS),
        // reshape to [384, 37, 56]
        std::vector<float> feats_3d[4];
        for (int f = 0; f < 4; f++) {
            // Apply final layer norm (same weights for all 4)
            std::vector<float> normed(SEQ_LEN * EMBED);
            layer_norm(block_outputs[f].data(), normed.data(),
                       w.norm_w.data(), w.norm_b.data(), SEQ_LEN, EMBED);

            // Extract patch tokens (skip CLS at position 0), reshape to [384, 37, 56]
            feats_3d[f].resize(EMBED * GRID_H * GRID_W);
            for (int p = 0; p < NUM_PATCHES; p++) {
                int h = p / GRID_W;
                int ww = p % GRID_W;
                for (int c = 0; c < EMBED; c++) {
                    feats_3d[f][c * GRID_H * GRID_W + h * GRID_W + ww] =
                        normed[(p + 1) * EMBED + c];
                }
            }
        }

        dpt_head_multi(feats_3d, output);
    }

private:
    void add_pos_embed(float* tokens) {
        // Original pos embed: [1, 1370, 384]
        // CLS pos embed is the first token
        for (int d = 0; d < EMBED; d++) {
            tokens[d] += w.pos_embed[d];
        }

        // Patch pos embeds: interpolate from [1, 1369, 384] -> [1, 2072, 384]
        // Original: sqrt(1369) = 37, so it's [37, 37, 384]
        // Target: [37, 56, 384]
        int orig_h = 37, orig_w = 37;

        // Bicubic interpolation of positional embeddings
        // Reshape [1369, 384] -> [384, 37, 37]
        std::vector<float> pos_2d(EMBED * orig_h * orig_w);
        for (int s = 0; s < orig_h * orig_w; s++) {
            int h = s / orig_w;
            int ww = s % orig_w;
            for (int c = 0; c < EMBED; c++) {
                pos_2d[c * orig_h * orig_w + h * orig_w + ww] = w.pos_embed[(s + 1) * EMBED + c];
            }
        }

        // Interpolate to [384, 37, 56] with align_corners=False (matches PyTorch graph)
        std::vector<float> pos_interp(EMBED * GRID_H * GRID_W);
        upsample_bicubic2d(pos_2d.data(), pos_interp.data(), EMBED, orig_h, orig_w, GRID_H, GRID_W, false);

        // Add to patch tokens
        for (int p = 0; p < NUM_PATCHES; p++) {
            int h = p / GRID_W;
            int ww = p % GRID_W;
            for (int c = 0; c < EMBED; c++) {
                tokens[(p + 1) * EMBED + c] += pos_interp[c * GRID_H * GRID_W + h * GRID_W + ww];
            }
        }
    }

    void transformer_block(float* tokens, float* tmp, float* tmp2,
                           const Weights::Block& blk) {
        // Pre-norm attention
        layer_norm(tokens, tmp, blk.norm1_w.data(), blk.norm1_b.data(), SEQ_LEN, EMBED);

        // Multi-head attention
        multihead_attention(tmp, tmp2, blk);

        // Residual + LayerScale
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < EMBED; d++)
                tokens[s * EMBED + d] += tmp2[s * EMBED + d] * blk.ls1_gamma[d];

        // Pre-norm MLP
        layer_norm(tokens, tmp, blk.norm2_w.data(), blk.norm2_b.data(), SEQ_LEN, EMBED);

        // MLP
        mlp_forward(tmp, tmp2, blk);

        // Residual + LayerScale
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < EMBED; d++)
                tokens[s * EMBED + d] += tmp2[s * EMBED + d] * blk.ls2_gamma[d];
    }

    void multihead_attention(const float* input, float* output,
                             const Weights::Block& blk) {
        // QKV projection: [SEQ_LEN, 384] x [1152, 384]^T -> [SEQ_LEN, 1152]
        std::vector<float> qkv(SEQ_LEN * 3 * EMBED);
        matmul_add(input, blk.attn_qkv_w.data(), qkv.data(),
                   blk.attn_qkv_b.data(), SEQ_LEN, 3 * EMBED, EMBED);

        // Reshape to [3, NUM_HEADS, SEQ_LEN, HEAD_DIM] and process per-head
        std::vector<float> attn_out(SEQ_LEN * EMBED, 0.0f);

        for (int h = 0; h < NUM_HEADS; h++) {
            // Extract Q, K, V for this head
            std::vector<float> Q(SEQ_LEN * HEAD_DIM);
            std::vector<float> K(SEQ_LEN * HEAD_DIM);
            std::vector<float> V(SEQ_LEN * HEAD_DIM);

            for (int s = 0; s < SEQ_LEN; s++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    Q[s * HEAD_DIM + d] = qkv[s * (3 * EMBED) + 0 * EMBED + h * HEAD_DIM + d];
                    K[s * HEAD_DIM + d] = qkv[s * (3 * EMBED) + 1 * EMBED + h * HEAD_DIM + d];
                    V[s * HEAD_DIM + d] = qkv[s * (3 * EMBED) + 2 * EMBED + h * HEAD_DIM + d];
                }
            }

            // Attention scores: Q @ K^T / sqrt(HEAD_DIM)
            std::vector<float> scores(SEQ_LEN * SEQ_LEN);
            matmul(Q.data(), K.data(), scores.data(), SEQ_LEN, SEQ_LEN, HEAD_DIM, false, true);

            float scale = 1.0f / std::sqrt((float)HEAD_DIM);
            for (int i = 0; i < SEQ_LEN * SEQ_LEN; i++) scores[i] *= scale;

            // Softmax
            softmax_inplace(scores.data(), SEQ_LEN, SEQ_LEN);

            // Attention @ V
            std::vector<float> head_out(SEQ_LEN * HEAD_DIM);
            matmul(scores.data(), V.data(), head_out.data(), SEQ_LEN, HEAD_DIM, SEQ_LEN);

            // Scatter back
            for (int s = 0; s < SEQ_LEN; s++)
                for (int d = 0; d < HEAD_DIM; d++)
                    attn_out[s * EMBED + h * HEAD_DIM + d] = head_out[s * HEAD_DIM + d];
        }

        // Output projection
        matmul_add(attn_out.data(), blk.attn_proj_w.data(), output,
                   blk.attn_proj_b.data(), SEQ_LEN, EMBED, EMBED);
    }

    void mlp_forward(const float* input, float* output, const Weights::Block& blk) {
        std::vector<float> hidden(SEQ_LEN * MLP_DIM);
        matmul_add(input, blk.mlp_fc1_w.data(), hidden.data(),
                   blk.mlp_fc1_b.data(), SEQ_LEN, MLP_DIM, EMBED);
        gelu_inplace(hidden.data(), SEQ_LEN * MLP_DIM);
        matmul_add(hidden.data(), blk.mlp_fc2_w.data(), output,
                   blk.mlp_fc2_b.data(), SEQ_LEN, EMBED, MLP_DIM);
    }

    // ResidualConvUnit
    void residual_conv_unit(const float* input, float* output,
                            const float* conv1_w, const float* conv1_b,
                            const float* conv2_w, const float* conv2_b,
                            int channels, int h, int ww) {
        int spatial = channels * h * ww;

        // ReLU + Conv1
        std::vector<float> tmp(spatial);
        for (int i = 0; i < spatial; i++) tmp[i] = std::max(0.0f, input[i]);

        std::vector<float> conv_out(spatial);
        conv2d(tmp.data(), conv_out.data(), conv1_w, conv1_b,
               channels, h, ww, channels, 3, 3, 1, 1, 1, 1);

        // ReLU + Conv2
        for (int i = 0; i < spatial; i++) tmp[i] = std::max(0.0f, conv_out[i]);
        conv2d(tmp.data(), conv_out.data(), conv2_w, conv2_b,
               channels, h, ww, channels, 3, 3, 1, 1, 1, 1);

        // Residual
        for (int i = 0; i < spatial; i++) output[i] = input[i] + conv_out[i];
    }

    void dpt_head_multi(std::vector<float> feats_3d[4], float* output) {
        // feats_3d[0..3] are [384, 37, 56] from blocks 2, 5, 8, 11 respectively
        // Each gets projected to a different scale

        // Scale 0 (block 2): 384->48, then ConvTranspose(48,48,4,stride=4) -> [48, 148, 224]
        std::vector<float> s0_proj(48 * GRID_H * GRID_W);
        conv2d(feats_3d[0].data(), s0_proj.data(), w.dpt.proj0_w.data(), w.dpt.proj0_b.data(),
               EMBED, GRID_H, GRID_W, 48, 1, 1, 1, 1, 0, 0);
        std::vector<float> s0(48 * 148 * 224);
        conv_transpose2d(s0_proj.data(), s0.data(), w.dpt.resize0_w.data(), w.dpt.resize0_b.data(),
                         48, GRID_H, GRID_W, 48, 4, 4, 4, 4, 0, 0);

        // Scale 1 (block 5): 384->96, then ConvTranspose(96,96,2,stride=2) -> [96, 74, 112]
        std::vector<float> s1_proj(96 * GRID_H * GRID_W);
        conv2d(feats_3d[1].data(), s1_proj.data(), w.dpt.proj1_w.data(), w.dpt.proj1_b.data(),
               EMBED, GRID_H, GRID_W, 96, 1, 1, 1, 1, 0, 0);
        std::vector<float> s1(96 * 74 * 112);
        conv_transpose2d(s1_proj.data(), s1.data(), w.dpt.resize1_w.data(), w.dpt.resize1_b.data(),
                         96, GRID_H, GRID_W, 96, 2, 2, 2, 2, 0, 0);

        // Scale 2 (block 8): 384->192 -> [192, 37, 56]
        std::vector<float> s2(192 * GRID_H * GRID_W);
        conv2d(feats_3d[2].data(), s2.data(), w.dpt.proj2_w.data(), w.dpt.proj2_b.data(),
               EMBED, GRID_H, GRID_W, 192, 1, 1, 1, 1, 0, 0);

        // Scale 3 (block 11): 384->384, then Conv(384,384,3,stride=2,pad=1) -> [384, 19, 28]
        std::vector<float> s3_proj(EMBED * GRID_H * GRID_W);
        conv2d(feats_3d[3].data(), s3_proj.data(), w.dpt.proj3_w.data(), w.dpt.proj3_b.data(),
               EMBED, GRID_H, GRID_W, EMBED, 1, 1, 1, 1, 0, 0);
        std::vector<float> s3(EMBED * 19 * 28);
        conv2d(s3_proj.data(), s3.data(), w.dpt.resize3_w.data(), w.dpt.resize3_b.data(),
               EMBED, GRID_H, GRID_W, EMBED, 3, 3, 2, 2, 1, 1);

        // layer_rn: 3x3 conv (no bias) to 64 channels
        std::vector<float> f1(64 * 148 * 224);
        conv2d_nobias(s0.data(), f1.data(), w.dpt.layer1_rn_w.data(),
                      48, 148, 224, 64, 3, 3, 1, 1, 1, 1);

        std::vector<float> f2(64 * 74 * 112);
        conv2d_nobias(s1.data(), f2.data(), w.dpt.layer2_rn_w.data(),
                      96, 74, 112, 64, 3, 3, 1, 1, 1, 1);

        std::vector<float> f3(64 * GRID_H * GRID_W);
        conv2d_nobias(s2.data(), f3.data(), w.dpt.layer3_rn_w.data(),
                      192, GRID_H, GRID_W, 64, 3, 3, 1, 1, 1, 1);

        std::vector<float> f4(64 * 19 * 28);
        conv2d_nobias(s3.data(), f4.data(), w.dpt.layer4_rn_w.data(),
                      EMBED, 19, 28, 64, 3, 3, 1, 1, 1, 1);

        // ===== RefineNet (corrected flow from graph trace) =====
        // Stage 4: rcu2(f4) -> upsample -> out_conv
        // Graph: relu(f4) -> conv -> relu -> conv -> add(result, f4) -> upsample -> out_conv
        std::vector<float> r4(64 * 19 * 28);
        auto& rs4 = w.dpt.refine[3];
        residual_conv_unit(f4.data(), r4.data(),
                           rs4.rcu2_conv1_w.data(), rs4.rcu2_conv1_b.data(),
                           rs4.rcu2_conv2_w.data(), rs4.rcu2_conv2_b.data(),
                           64, 19, 28);
        // Upsample FIRST, then out_conv
        std::vector<float> r4_up(64 * GRID_H * GRID_W);
        upsample_bilinear2d(r4.data(), r4_up.data(), 64, 19, 28, GRID_H, GRID_W);
        std::vector<float> r4_out(64 * GRID_H * GRID_W);
        conv2d(r4_up.data(), r4_out.data(), rs4.out_conv_w.data(), rs4.out_conv_b.data(),
               64, GRID_H, GRID_W, 64, 1, 1, 1, 1, 0, 0);

        // Stage 3: rcu1(f3) -> merge(out4, rcu1_result) -> rcu2 -> upsample -> out_conv
        int sz3 = 64 * GRID_H * GRID_W;
        auto& rs3 = w.dpt.refine[2];
        // RCU1 on f3 alone
        std::vector<float> r3_rcu1(sz3);
        residual_conv_unit(f3.data(), r3_rcu1.data(),
                           rs3.rcu1_conv1_w.data(), rs3.rcu1_conv1_b.data(),
                           rs3.rcu1_conv2_w.data(), rs3.rcu1_conv2_b.data(),
                           64, GRID_H, GRID_W);
        // Merge: out_conv(stage4) + rcu1(f3)
        std::vector<float> merged3(sz3);
        for (int i = 0; i < sz3; i++) merged3[i] = r4_out[i] + r3_rcu1[i];
        // RCU2 on merged
        std::vector<float> r3b(sz3);
        residual_conv_unit(merged3.data(), r3b.data(),
                           rs3.rcu2_conv1_w.data(), rs3.rcu2_conv1_b.data(),
                           rs3.rcu2_conv2_w.data(), rs3.rcu2_conv2_b.data(),
                           64, GRID_H, GRID_W);
        // Upsample then out_conv
        std::vector<float> r3_up(64 * 74 * 112);
        upsample_bilinear2d(r3b.data(), r3_up.data(), 64, GRID_H, GRID_W, 74, 112);
        std::vector<float> r3_out(64 * 74 * 112);
        conv2d(r3_up.data(), r3_out.data(), rs3.out_conv_w.data(), rs3.out_conv_b.data(),
               64, 74, 112, 64, 1, 1, 1, 1, 0, 0);

        // Stage 2: rcu1(f2) -> merge(out3, rcu1_result) -> rcu2 -> upsample -> out_conv
        int sz2 = 64 * 74 * 112;
        auto& rs2 = w.dpt.refine[1];
        std::vector<float> r2_rcu1(sz2);
        residual_conv_unit(f2.data(), r2_rcu1.data(),
                           rs2.rcu1_conv1_w.data(), rs2.rcu1_conv1_b.data(),
                           rs2.rcu1_conv2_w.data(), rs2.rcu1_conv2_b.data(),
                           64, 74, 112);
        std::vector<float> merged2(sz2);
        for (int i = 0; i < sz2; i++) merged2[i] = r3_out[i] + r2_rcu1[i];
        std::vector<float> r2b(sz2);
        residual_conv_unit(merged2.data(), r2b.data(),
                           rs2.rcu2_conv1_w.data(), rs2.rcu2_conv1_b.data(),
                           rs2.rcu2_conv2_w.data(), rs2.rcu2_conv2_b.data(),
                           64, 74, 112);
        std::vector<float> r2_up(64 * 148 * 224);
        upsample_bilinear2d(r2b.data(), r2_up.data(), 64, 74, 112, 148, 224);
        std::vector<float> r2_out(64 * 148 * 224);
        conv2d(r2_up.data(), r2_out.data(), rs2.out_conv_w.data(), rs2.out_conv_b.data(),
               64, 148, 224, 64, 1, 1, 1, 1, 0, 0);

        // Stage 1: rcu1(f1) -> merge(out2, rcu1_result) -> rcu2 -> upsample(2x) -> out_conv
        int sz1 = 64 * 148 * 224;
        auto& rs1 = w.dpt.refine[0];
        std::vector<float> r1_rcu1(sz1);
        residual_conv_unit(f1.data(), r1_rcu1.data(),
                           rs1.rcu1_conv1_w.data(), rs1.rcu1_conv1_b.data(),
                           rs1.rcu1_conv2_w.data(), rs1.rcu1_conv2_b.data(),
                           64, 148, 224);
        std::vector<float> merged1(sz1);
        for (int i = 0; i < sz1; i++) merged1[i] = r2_out[i] + r1_rcu1[i];
        std::vector<float> r1b(sz1);
        residual_conv_unit(merged1.data(), r1b.data(),
                           rs1.rcu2_conv1_w.data(), rs1.rcu2_conv1_b.data(),
                           rs1.rcu2_conv2_w.data(), rs1.rcu2_conv2_b.data(),
                           64, 148, 224);
        // Upsample 2x: [148,224] -> [296,448]
        std::vector<float> r1_up(64 * 296 * 448);
        upsample_bilinear2d(r1b.data(), r1_up.data(), 64, 148, 224, 296, 448);
        std::vector<float> r1_out(64 * 296 * 448);
        conv2d(r1_up.data(), r1_out.data(), rs1.out_conv_w.data(), rs1.out_conv_b.data(),
               64, 296, 448, 64, 1, 1, 1, 1, 0, 0);

        // Output head
        // Conv(64->32, 3x3, pad=1)
        std::vector<float> head1(32 * 296 * 448);
        conv2d(r1_out.data(), head1.data(), w.dpt.out_conv1_w.data(), w.dpt.out_conv1_b.data(),
               64, 296, 448, 32, 3, 3, 1, 1, 1, 1);

        // Upsample [32, 296, 448] -> [32, 518, 784]
        // Note: output size matches input size exactly
        std::vector<float> head1_up(32 * OUT_H * OUT_W);
        upsample_bilinear2d(head1.data(), head1_up.data(), 32, 296, 448, OUT_H, OUT_W);

        // Conv(32->32, 3x3, pad=1) + ReLU
        std::vector<float> head2(32 * OUT_H * OUT_W);
        conv2d(head1_up.data(), head2.data(), w.dpt.out_conv2_0_w.data(), w.dpt.out_conv2_0_b.data(),
               32, OUT_H, OUT_W, 32, 3, 3, 1, 1, 1, 1);
        relu_inplace(head2.data(), 32 * OUT_H * OUT_W);

        // Conv(32->1, 1x1) + ReLU
        std::vector<float> depth(1 * OUT_H * OUT_W);
        conv2d(head2.data(), depth.data(), w.dpt.out_conv2_2_w.data(), w.dpt.out_conv2_2_b.data(),
               32, OUT_H, OUT_W, 1, 1, 1, 1, 1, 0, 0);
        relu_inplace(depth.data(), OUT_H * OUT_W);

        // Output: squeeze [1, 1, 518, 784] -> [1, 518, 784]
        std::memcpy(output, depth.data(), OUT_H * OUT_W * sizeof(float));
    }
};

} // namespace dav2
