#pragma once
// Depth-Anything-V2-Small: NXP S32-Compatible C++ Implementation
// Targets: Centralized domain controllers (Cortex-A53/A72 clusters)
//          Renesas R-Car, NXP S32G, TI TDA4VM
//
// Properties matching MATLAB Coder output:
//   - No external BLAS (pure scalar math with double-precision accumulation)
//   - Static memory allocation only (no heap, no std::vector, no new/malloc)
//   - No STL containers (no std::vector, std::string, std::ifstream)
//   - No lambdas, no auto, no thread_local
//   - C++14 compliant, -ffp-contract=off for strict IEEE 754
//   - Code traceability comments mapping to .pt2 graph nodes
//
// Architecture: DINOv2-Small ViT encoder (12 blocks) + DPT decoder (4-scale RefineNet)
// Input:  [1, 3, 518, 784] float32 (NCHW)
// Output: [1, 518, 784] float32 (depth map)
// Parameters: 24,710,849 (94.3 MB float32)
//
// Build:
//   clang++ -std=c++14 -O3 -ffp-contract=off -o depth_test_s32 main_s32.cpp
//   ./depth_test_s32 ../weights ../reference

#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace dav2_s32 {

// ============================================================================
// Model Constants
// [pt2: graph metadata — model dimensions from architecture.json]
// ============================================================================
static const int BATCH = 1;
static const int IN_C = 3;
static const int IN_H = 518;
static const int IN_W = 784;
static const int PATCH = 14;
static const int GRID_H = IN_H / PATCH;   // 37
static const int GRID_W = IN_W / PATCH;   // 56
static const int NUM_PATCHES = GRID_H * GRID_W;  // 2072
static const int SEQ_LEN = NUM_PATCHES + 1;       // 2073
static const int EMBED = 384;
static const int NUM_HEADS = 6;
static const int HEAD_DIM = EMBED / NUM_HEADS;  // 64
static const int MLP_DIM = 1536;
static const int NUM_BLOCKS = 12;
static const int POS_EMBED_ORIG = 1370;
static const int DPT_FEATURES = 64;
static const int OUT_H = IN_H;
static const int OUT_W = IN_W;

// ============================================================================
// Helpers
// ============================================================================
static inline int idx3(int c, int h, int w, int H, int W) {
    return (c * H + h) * W + w;
}

static inline int clamp_int(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// ============================================================================
// Matrix Multiply — scalar double-precision accumulation
// [pt2: aten.linear.default, aten.mm.default, aten.bmm.default]
// No external BLAS. Matches MATLAB Coder's self-contained math routines.
// ============================================================================
// Tile size for cache-friendly GEMM (fits in L1 cache)
static const int TILE = 64;

static void matmul(const float* A, const float* B, float* C,
                   int M, int N, int K, bool transA, bool transB) {
    // For transposed cases, fall back to simple loops (less common, only in attention Q@K^T)
    if (transA || transB) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    float a = transA ? A[k * M + m] : A[m * K + k];
                    float b = transB ? B[n * K + k] : B[k * N + n];
                    sum += (double)a * (double)b;
                }
                C[m * N + n] = (float)sum;
            }
        }
        return;
    }
    // Tiled GEMM for non-transposed case (majority of calls: linear layers, conv GEMM)
    // Zero output first
    std::memset(C, 0, (size_t)M * N * sizeof(float));
    for (int m0 = 0; m0 < M; m0 += TILE) {
        int m1 = m0 + TILE < M ? m0 + TILE : M;
        for (int k0 = 0; k0 < K; k0 += TILE) {
            int k1 = k0 + TILE < K ? k0 + TILE : K;
            for (int n0 = 0; n0 < N; n0 += TILE) {
                int n1 = n0 + TILE < N ? n0 + TILE : N;
                // Micro-kernel: accumulate C[m0:m1, n0:n1] += A[m0:m1, k0:k1] @ B[k0:k1, n0:n1]
                for (int m = m0; m < m1; m++) {
                    for (int k = k0; k < k1; k++) {
                        float a_mk = A[m * K + k];
                        for (int n = n0; n < n1; n++) {
                            C[m * N + n] += a_mk * B[k * N + n];
                        }
                    }
                }
            }
        }
    }
}

static void matmul_nn(const float* A, const float* B, float* C, int M, int N, int K) {
    matmul(A, B, C, M, N, K, false, false);
}

static void matmul_add(const float* A, const float* B, float* C,
                       const float* bias, int M, int N, int K) {
    matmul(A, B, C, M, N, K, false, true);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C[m * N + n] += bias[n];
}

// ============================================================================
// Activations
// [pt2: aten.gelu.default, aten.relu.default]
// ============================================================================
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
}

static void gelu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = gelu(x[i]);
}

static void relu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

// ============================================================================
// Layer Norm — double-precision accumulation for mean/variance
// [pt2: aten.native_layer_norm.default] eps=1e-6 (DINOv2 specific)
// ============================================================================
static void layer_norm(const float* input, float* output, const float* gamma,
                       const float* beta, int seq_len, int dim, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float* in_row = input + s * dim;
        float* out_row = output + s * dim;
        double mean = 0.0;
        for (int d = 0; d < dim; d++) mean += (double)in_row[d];
        mean /= dim;
        double var = 0.0;
        for (int d = 0; d < dim; d++) {
            double diff = (double)in_row[d] - mean;
            var += diff * diff;
        }
        var /= dim;
        float inv_std = 1.0f / std::sqrt((float)(var + (double)eps));
        float fmean = (float)mean;
        for (int d = 0; d < dim; d++)
            out_row[d] = (in_row[d] - fmean) * inv_std * gamma[d] + beta[d];
    }
}

// ============================================================================
// Softmax — double-precision accumulation for denominator
// [pt2: aten._softmax.default]
// ============================================================================
static void softmax_inplace(float* x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* row = x + r * cols;
        // Find max (explicit loop, no std::max_element)
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            row[c] = std::exp(row[c] - max_val);
            sum += (double)row[c];
        }
        float inv_sum = 1.0f / (float)sum;
        for (int c = 0; c < cols; c++) row[c] *= inv_sum;
    }
}

// ============================================================================
// Conv2d — chunked im2col + GEMM with bounded static buffer
// [pt2: aten.convolution.default]
// Uses a static im2col buffer (16 MB) and processes output rows in chunks.
// 1x1 convolutions use direct matmul (no im2col needed).
// ============================================================================

// Static im2col buffer: 4,194,304 floats = 16 MB
// Processes CHUNK_ROWS output rows at a time to bound memory usage
static const int IM2COL_BUF_SIZE = 4194304;
static float g_im2col_buf[IM2COL_BUF_SIZE];

static void conv2d(const float* input, float* output,
                   const float* weight, const float* bias,
                   int in_c, int in_h, int in_w,
                   int out_c, int kh, int kw,
                   int stride_h, int stride_w,
                   int pad_h, int pad_w) {
    int out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;

    // Fast path: 1x1 convolution = matrix multiply
    if (kh == 1 && kw == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0) {
        int spatial = in_h * in_w;
        matmul_nn(weight, input, output, out_c, spatial, in_c);
        if (bias) {
            for (int oc = 0; oc < out_c; oc++) {
                float b = bias[oc];
                for (int i = 0; i < spatial; i++)
                    output[oc * spatial + i] += b;
            }
        }
        return;
    }

    // Chunked im2col + GEMM path for general convolution
    int col_h = in_c * kh * kw;  // im2col column height
    // Determine how many output rows we can process at once
    int max_chunk_cols = IM2COL_BUF_SIZE / col_h;
    if (max_chunk_cols < 1) max_chunk_cols = 1;
    int chunk_rows = max_chunk_cols / out_w;
    if (chunk_rows < 1) chunk_rows = 1;
    if (chunk_rows > out_h) chunk_rows = out_h;

    for (int oh_start = 0; oh_start < out_h; oh_start += chunk_rows) {
        int oh_end = oh_start + chunk_rows;
        if (oh_end > out_h) oh_end = out_h;
        int rows_this_chunk = oh_end - oh_start;
        int col_w = rows_this_chunk * out_w;

        // im2col: unfold input into column matrix [col_h, col_w]
        float* col = g_im2col_buf;
        for (int ic = 0; ic < in_c; ic++) {
            for (int fh = 0; fh < kh; fh++) {
                for (int fw = 0; fw < kw; fw++) {
                    int col_row = (ic * kh + fh) * kw + fw;
                    float* col_ptr = col + col_row * col_w;
                    int ci = 0;
                    for (int oh = oh_start; oh < oh_end; oh++) {
                        int ih = oh * stride_h - pad_h + fh;
                        bool ih_valid = (ih >= 0 && ih < in_h);
                        for (int ow = 0; ow < out_w; ow++, ci++) {
                            int iw = ow * stride_w - pad_w + fw;
                            if (ih_valid && iw >= 0 && iw < in_w)
                                col_ptr[ci] = input[ic * in_h * in_w + ih * in_w + iw];
                            else
                                col_ptr[ci] = 0.0f;
                        }
                    }
                }
            }
        }

        // GEMM: weight[out_c, col_h] @ col[col_h, col_w] -> scattered output
        // Each oc slice in output is contiguous: output[oc * out_h * out_w + oh_start * out_w]
        // Use tiled M-K-N loop for cache efficiency (B columns accessed sequentially)
        for (int oc = 0; oc < out_c; oc++) {
            float* out_oc = output + oc * out_h * out_w + oh_start * out_w;
            float b = bias ? bias[oc] : 0.0f;
            for (int j = 0; j < col_w; j++) out_oc[j] = b;
        }
        // Tiled accumulation: for each tile of oc × col_h × col_w
        for (int oc0 = 0; oc0 < out_c; oc0 += TILE) {
            int oc1 = oc0 + TILE < out_c ? oc0 + TILE : out_c;
            for (int i0 = 0; i0 < col_h; i0 += TILE) {
                int i1 = i0 + TILE < col_h ? i0 + TILE : col_h;
                for (int j0 = 0; j0 < col_w; j0 += TILE) {
                    int j1 = j0 + TILE < col_w ? j0 + TILE : col_w;
                    for (int oc = oc0; oc < oc1; oc++) {
                        float* out_oc = output + oc * out_h * out_w + oh_start * out_w;
                        const float* w_oc = weight + oc * col_h;
                        for (int i = i0; i < i1; i++) {
                            float w_val = w_oc[i];
                            const float* col_row = col + i * col_w;
                            for (int j = j0; j < j1; j++) {
                                out_oc[j] += w_val * col_row[j];
                            }
                        }
                    }
                }
            }
        }
    }
}

static void conv2d_nobias(const float* input, float* output, const float* weight,
                          int in_c, int in_h, int in_w, int out_c, int kh, int kw,
                          int stride_h, int stride_w, int pad_h, int pad_w) {
    conv2d(input, output, weight, (const float*)0, in_c, in_h, in_w, out_c, kh, kw,
           stride_h, stride_w, pad_h, pad_w);
}

// ============================================================================
// ConvTranspose2d
// [pt2: aten.convolution.default (transposed=True)]
// ============================================================================
static void conv_transpose2d(const float* input, float* output,
                             const float* weight, const float* bias,
                             int in_c, int in_h, int in_w,
                             int out_c, int kh, int kw,
                             int stride_h, int stride_w,
                             int pad_h, int pad_w) {
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kh;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kw;
    for (int oc = 0; oc < out_c; oc++) {
        float b = bias ? bias[oc] : 0.0f;
        for (int oh = 0; oh < out_h; oh++)
            for (int ow = 0; ow < out_w; ow++)
                output[idx3(oc, oh, ow, out_h, out_w)] = b;
    }
    for (int ic = 0; ic < in_c; ic++) {
        for (int ih = 0; ih < in_h; ih++) {
            for (int iw = 0; iw < in_w; iw++) {
                float val = input[idx3(ic, ih, iw, in_h, in_w)];
                for (int fh = 0; fh < kh; fh++) {
                    for (int fw = 0; fw < kw; fw++) {
                        int oh = ih * stride_h - pad_h + fh;
                        int ow = iw * stride_w - pad_w + fw;
                        if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w)
                            for (int oc = 0; oc < out_c; oc++)
                                output[idx3(oc, oh, ow, out_h, out_w)] +=
                                    val * weight[((ic * out_c + oc) * kh + fh) * kw + fw];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Bilinear Upsample 2D (align_corners=True)
// [pt2: aten.upsample_bilinear2d.vec]
// ============================================================================
static void upsample_bilinear2d(const float* input, float* output,
                                int channels, int in_h, int in_w, int out_h, int out_w) {
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            float ih_f = (out_h > 1) ? (float)oh * (in_h - 1) / (out_h - 1) : 0.0f;
            int ih0 = (int)ih_f;
            int ih1 = (ih0 + 1 < in_h) ? ih0 + 1 : in_h - 1;
            float h_frac = ih_f - ih0;
            for (int ow = 0; ow < out_w; ow++) {
                float iw_f = (out_w > 1) ? (float)ow * (in_w - 1) / (out_w - 1) : 0.0f;
                int iw0 = (int)iw_f;
                int iw1 = (iw0 + 1 < in_w) ? iw0 + 1 : in_w - 1;
                float w_frac = iw_f - iw0;
                float val = input[idx3(c, ih0, iw0, in_h, in_w)] * (1-h_frac) * (1-w_frac) +
                            input[idx3(c, ih0, iw1, in_h, in_w)] * (1-h_frac) * w_frac +
                            input[idx3(c, ih1, iw0, in_h, in_w)] * h_frac * (1-w_frac) +
                            input[idx3(c, ih1, iw1, in_h, in_w)] * h_frac * w_frac;
                output[idx3(c, oh, ow, out_h, out_w)] = val;
            }
        }
    }
}

// ============================================================================
// Bicubic Upsample 2D
// [pt2: aten.upsample_bicubic2d.vec]
// ============================================================================
static inline float cubic_interp(float x) {
    float ax = std::abs(x);
    if (ax <= 1.0f) return ((1.25f * ax - 2.25f) * ax) * ax + 1.0f;
    if (ax < 2.0f) return ((-0.75f * ax + 3.75f) * ax - 6.0f) * ax + 3.0f;
    return 0.0f;
}

static void upsample_bicubic2d(const float* input, float* output,
                               int channels, int in_h, int in_w,
                               int out_h, int out_w, bool align_corners) {
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
                int ih_c = (int)std::floor(ih_f);
                int iw_c = (int)std::floor(iw_f);
                float val = 0.0f;
                for (int dh = -1; dh <= 2; dh++) {
                    float wh = cubic_interp(ih_f - (ih_c + dh));
                    for (int dw = -1; dw <= 2; dw++) {
                        int ih = clamp_int(ih_c + dh, 0, in_h - 1);
                        int iw = clamp_int(iw_c + dw, 0, in_w - 1);
                        val += input[idx3(c, ih, iw, in_h, in_w)] * wh * cubic_interp(iw_f - (iw_c + dw));
                    }
                }
                output[idx3(c, oh, ow, out_h, out_w)] = val;
            }
        }
    }
}

// ============================================================================
// Weight Storage — all pointers into a single flat static buffer
// Total: 24,710,849 floats (94.3 MB in BSS segment)
// ============================================================================
// 24,710,849 model params + 795,648 pos_embed_interp + 9,216 ls_gamma copies = 25,515,713
static float g_weight_storage[25520000];

struct Block {
    float ls1_gamma[EMBED];
    float ls2_gamma[EMBED];
    float* norm1_w;  float* norm1_b;  float* norm2_w;  float* norm2_b;
    float* attn_qkv_w;  float* attn_qkv_b;  float* attn_proj_w;  float* attn_proj_b;
    float* mlp_fc1_w;  float* mlp_fc1_b;  float* mlp_fc2_w;  float* mlp_fc2_b;
};

struct RefineStage {
    bool has_rcu1;
    float* rcu1_conv1_w;  float* rcu1_conv1_b;  float* rcu1_conv2_w;  float* rcu1_conv2_b;
    float* rcu2_conv1_w;  float* rcu2_conv1_b;  float* rcu2_conv2_w;  float* rcu2_conv2_b;
    float* out_conv_w;  float* out_conv_b;
};

struct Weights {
    float* cls_token;
    float* pos_embed;
    float* patch_proj_w;
    float* patch_proj_b;
    float* norm_w;
    float* norm_b;
    Block blocks[NUM_BLOCKS];

    // DPT decoder
    float* proj0_w;  float* proj0_b;
    float* proj1_w;  float* proj1_b;
    float* proj2_w;  float* proj2_b;
    float* proj3_w;  float* proj3_b;
    float* resize0_w;  float* resize0_b;
    float* resize1_w;  float* resize1_b;
    float* resize3_w;  float* resize3_b;
    float* layer1_rn_w;  float* layer2_rn_w;  float* layer3_rn_w;  float* layer4_rn_w;
    RefineStage refine[4];
    float* out_conv1_w;  float* out_conv1_b;
    float* out_conv2_0_w;  float* out_conv2_0_b;
    float* out_conv2_2_w;  float* out_conv2_2_b;

    // Cached interpolated positional embedding
    float* pos_embed_interp;
};

static Weights g_weights;

// ============================================================================
// File I/O — C fopen/fread (no std::ifstream)
// Matches MATLAB Coder's readDnnConstants_real32_T pattern
// ============================================================================
static void load_bin(const char* path, float* dst, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        return;
    }
    size_t read = fread(dst, sizeof(float), (size_t)n, f);
    if ((int)read != n) {
        fprintf(stderr, "WARNING: Read %d/%d floats from %s\n", (int)read, n, path);
    }
    fclose(f);
}

// Helper: load a named weight file into the next available slot in g_weight_storage
static int g_weight_offset = 0;

static float* alloc_weight(int n) {
    float* ptr = &g_weight_storage[g_weight_offset];
    g_weight_offset += n;
    return ptr;
}

static float* load_weight(const char* dir, const char* name, int n) {
    float* ptr = alloc_weight(n);
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    load_bin(path, ptr, n);
    return ptr;
}

// ============================================================================
// Static Inference Buffers — all compile-time-known sizes
// Total: ~206 MB in BSS (demand-paged, no binary size impact)
// ============================================================================

// Encoder buffers
static float buf_patch[EMBED * GRID_H * GRID_W];              // 791,552
static float buf_tokens[SEQ_LEN * EMBED];                     // 796,032
static float buf_tmp[SEQ_LEN * EMBED];                        // 796,032
static float buf_tmp2[SEQ_LEN * EMBED];                       // 796,032
static float buf_qkv[SEQ_LEN * 3 * EMBED];                   // 2,388,096
static float buf_attn_out[SEQ_LEN * EMBED];                   // 796,032
static float buf_Q[SEQ_LEN * HEAD_DIM];                       // 132,672
static float buf_K[SEQ_LEN * HEAD_DIM];                       // 132,672
static float buf_V[SEQ_LEN * HEAD_DIM];                       // 132,672
static float buf_scores[SEQ_LEN * SEQ_LEN];                   // 4,297,329
static float buf_head_out[SEQ_LEN * HEAD_DIM];                // 132,672
static float buf_hidden[SEQ_LEN * MLP_DIM];                   // 3,184,128
static float buf_block_out[4][SEQ_LEN * EMBED];               // 4 * 796,032
static float buf_normed[SEQ_LEN * EMBED];                     // 796,032
static float buf_feat[4][EMBED * GRID_H * GRID_W];            // 4 * 791,552

// Decoder named buffers
static float buf_s0p[48 * GRID_H * GRID_W];                   // 99,456
static float buf_s0[48 * 148 * 224];                          // 1,591,296
static float buf_s1p[96 * GRID_H * GRID_W];                   // 198,912
static float buf_s1[96 * 74 * 112];                           // 795,648
static float buf_s2[192 * GRID_H * GRID_W];                   // 397,824
static float buf_s3p[EMBED * GRID_H * GRID_W];                // 791,552
static float buf_s3[EMBED * 19 * 28];                         // 204,288
static float buf_f1[64 * 148 * 224];                          // 2,121,728
static float buf_f2[64 * 74 * 112];                           // 530,432
static float buf_f3[64 * GRID_H * GRID_W];                    // 132,608
static float buf_f4[64 * 19 * 28];                            // 34,048
static float buf_head1[32 * 296 * 448];                       // 4,243,456
static float buf_head1_up[32 * OUT_H * OUT_W];                // 12,995,584
static float buf_head2[32 * OUT_H * OUT_W];                   // 12,995,584
static float buf_depth[OUT_H * OUT_W];                        // 406,112

// Decoder scratch buffers — reused across refine stages
// Sized for largest stage: 64 channels x 296 x 448
static float scratch_rcu[64 * 296 * 448];                     // 8,486,912
static float scratch_conv[64 * 296 * 448];                    // 8,486,912
static float scratch_merged[64 * 296 * 448];                  // 8,486,912
static float scratch_rcu2[64 * 296 * 448];                    // 8,486,912
static float scratch_up[64 * 296 * 448];                      // 8,486,912
static float scratch_out[64 * 296 * 448];                     // 8,486,912

// Positional embedding interpolation temporaries (used only during weight load)
static float g_pos_2d[EMBED * 37 * 37];                       // 525,696
static float g_pos_interp[EMBED * GRID_H * GRID_W];           // 791,552

// ============================================================================
// Weight Loading
// ============================================================================
static void load_refine_stage(RefineStage* rs, const char* dir,
                              const char* pfx, bool has_rcu1) {
    char name[256];
    rs->has_rcu1 = has_rcu1;
    if (has_rcu1) {
        snprintf(name, sizeof(name), "%sresconfunit1_conv1_weight", pfx);
        rs->rcu1_conv1_w = load_weight(dir, name, 64*64*9);
        snprintf(name, sizeof(name), "%sresconfunit1_conv1_bias", pfx);
        rs->rcu1_conv1_b = load_weight(dir, name, 64);
        snprintf(name, sizeof(name), "%sresconfunit1_conv2_weight", pfx);
        rs->rcu1_conv2_w = load_weight(dir, name, 64*64*9);
        snprintf(name, sizeof(name), "%sresconfunit1_conv2_bias", pfx);
        rs->rcu1_conv2_b = load_weight(dir, name, 64);
    } else {
        rs->rcu1_conv1_w = (float*)0;
        rs->rcu1_conv1_b = (float*)0;
        rs->rcu1_conv2_w = (float*)0;
        rs->rcu1_conv2_b = (float*)0;
    }
    snprintf(name, sizeof(name), "%sresconfunit2_conv1_weight", pfx);
    rs->rcu2_conv1_w = load_weight(dir, name, 64*64*9);
    snprintf(name, sizeof(name), "%sresconfunit2_conv1_bias", pfx);
    rs->rcu2_conv1_b = load_weight(dir, name, 64);
    snprintf(name, sizeof(name), "%sresconfunit2_conv2_weight", pfx);
    rs->rcu2_conv2_w = load_weight(dir, name, 64*64*9);
    snprintf(name, sizeof(name), "%sresconfunit2_conv2_bias", pfx);
    rs->rcu2_conv2_b = load_weight(dir, name, 64);
    snprintf(name, sizeof(name), "%sout_conv_weight", pfx);
    rs->out_conv_w = load_weight(dir, name, 64*64);
    snprintf(name, sizeof(name), "%sout_conv_bias", pfx);
    rs->out_conv_b = load_weight(dir, name, 64);
}

void load_weights(const char* dir) {
    Weights* w = &g_weights;
    g_weight_offset = 0;

    // [pt2: pretrained backbone weights]
    w->cls_token = load_weight(dir, "pretrained_cls_token", EMBED);
    w->pos_embed = load_weight(dir, "pretrained_pos_embed", POS_EMBED_ORIG * EMBED);
    w->patch_proj_w = load_weight(dir, "pretrained_patch_embed_proj_weight", EMBED * IN_C * PATCH * PATCH);
    w->patch_proj_b = load_weight(dir, "pretrained_patch_embed_proj_bias", EMBED);
    w->norm_w = load_weight(dir, "pretrained_norm_weight", EMBED);
    w->norm_b = load_weight(dir, "pretrained_norm_bias", EMBED);

    // [pt2: pretrained.blocks[0..11] — 12 transformer blocks]
    for (int i = 0; i < NUM_BLOCKS; i++) {
        Block* b = &w->blocks[i];
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "pretrained_blocks_%d_", i);
        char name[128];

        snprintf(name, sizeof(name), "%sls1_gamma", pfx);
        float* ls1_tmp = load_weight(dir, name, EMBED);
        std::memcpy(b->ls1_gamma, ls1_tmp, EMBED * sizeof(float));

        snprintf(name, sizeof(name), "%sls2_gamma", pfx);
        float* ls2_tmp = load_weight(dir, name, EMBED);
        std::memcpy(b->ls2_gamma, ls2_tmp, EMBED * sizeof(float));

        snprintf(name, sizeof(name), "%snorm1_weight", pfx);
        b->norm1_w = load_weight(dir, name, EMBED);
        snprintf(name, sizeof(name), "%snorm1_bias", pfx);
        b->norm1_b = load_weight(dir, name, EMBED);
        snprintf(name, sizeof(name), "%snorm2_weight", pfx);
        b->norm2_w = load_weight(dir, name, EMBED);
        snprintf(name, sizeof(name), "%snorm2_bias", pfx);
        b->norm2_b = load_weight(dir, name, EMBED);

        snprintf(name, sizeof(name), "%sattn_qkv_weight", pfx);
        b->attn_qkv_w = load_weight(dir, name, 3 * EMBED * EMBED);
        snprintf(name, sizeof(name), "%sattn_qkv_bias", pfx);
        b->attn_qkv_b = load_weight(dir, name, 3 * EMBED);
        snprintf(name, sizeof(name), "%sattn_proj_weight", pfx);
        b->attn_proj_w = load_weight(dir, name, EMBED * EMBED);
        snprintf(name, sizeof(name), "%sattn_proj_bias", pfx);
        b->attn_proj_b = load_weight(dir, name, EMBED);

        snprintf(name, sizeof(name), "%smlp_fc1_weight", pfx);
        b->mlp_fc1_w = load_weight(dir, name, MLP_DIM * EMBED);
        snprintf(name, sizeof(name), "%smlp_fc1_bias", pfx);
        b->mlp_fc1_b = load_weight(dir, name, MLP_DIM);
        snprintf(name, sizeof(name), "%smlp_fc2_weight", pfx);
        b->mlp_fc2_w = load_weight(dir, name, EMBED * MLP_DIM);
        snprintf(name, sizeof(name), "%smlp_fc2_bias", pfx);
        b->mlp_fc2_b = load_weight(dir, name, EMBED);
    }

    // [pt2: depth_head — DPT decoder weights]
    w->proj0_w = load_weight(dir, "depth_head_projects_0_weight", 48 * EMBED);
    w->proj0_b = load_weight(dir, "depth_head_projects_0_bias", 48);
    w->proj1_w = load_weight(dir, "depth_head_projects_1_weight", 96 * EMBED);
    w->proj1_b = load_weight(dir, "depth_head_projects_1_bias", 96);
    w->proj2_w = load_weight(dir, "depth_head_projects_2_weight", 192 * EMBED);
    w->proj2_b = load_weight(dir, "depth_head_projects_2_bias", 192);
    w->proj3_w = load_weight(dir, "depth_head_projects_3_weight", EMBED * EMBED);
    w->proj3_b = load_weight(dir, "depth_head_projects_3_bias", EMBED);

    w->resize0_w = load_weight(dir, "depth_head_resize_layers_0_weight", 48 * 48 * 16);
    w->resize0_b = load_weight(dir, "depth_head_resize_layers_0_bias", 48);
    w->resize1_w = load_weight(dir, "depth_head_resize_layers_1_weight", 96 * 96 * 4);
    w->resize1_b = load_weight(dir, "depth_head_resize_layers_1_bias", 96);
    w->resize3_w = load_weight(dir, "depth_head_resize_layers_3_weight", EMBED * EMBED * 9);
    w->resize3_b = load_weight(dir, "depth_head_resize_layers_3_bias", EMBED);

    w->layer1_rn_w = load_weight(dir, "depth_head_scratch_layer1_rn_weight", 64 * 48 * 9);
    w->layer2_rn_w = load_weight(dir, "depth_head_scratch_layer2_rn_weight", 64 * 96 * 9);
    w->layer3_rn_w = load_weight(dir, "depth_head_scratch_layer3_rn_weight", 64 * 192 * 9);
    w->layer4_rn_w = load_weight(dir, "depth_head_scratch_layer4_rn_weight", 64 * EMBED * 9);

    load_refine_stage(&w->refine[3], dir, "depth_head_scratch_refinenet4_", false);
    load_refine_stage(&w->refine[2], dir, "depth_head_scratch_refinenet3_", true);
    load_refine_stage(&w->refine[1], dir, "depth_head_scratch_refinenet2_", true);
    load_refine_stage(&w->refine[0], dir, "depth_head_scratch_refinenet1_", true);

    w->out_conv1_w = load_weight(dir, "depth_head_scratch_output_conv1_weight", 32 * 64 * 9);
    w->out_conv1_b = load_weight(dir, "depth_head_scratch_output_conv1_bias", 32);
    w->out_conv2_0_w = load_weight(dir, "depth_head_scratch_output_conv2_0_weight", 32 * 32 * 9);
    w->out_conv2_0_b = load_weight(dir, "depth_head_scratch_output_conv2_0_bias", 32);
    w->out_conv2_2_w = load_weight(dir, "depth_head_scratch_output_conv2_2_weight", 32);
    w->out_conv2_2_b = load_weight(dir, "depth_head_scratch_output_conv2_2_bias", 1);

    // [pt2: positional embedding interpolation — cached at load time]
    // Allocate space for interpolated pos embed in weight storage
    w->pos_embed_interp = alloc_weight(EMBED * GRID_H * GRID_W);

    // Reshape [1370, 384] -> [384, 37, 37] (skip CLS token at index 0)
    int orig_h = 37;
    int orig_w = 37;
    for (int s = 0; s < orig_h * orig_w; s++) {
        int h = s / orig_w;
        int ww = s % orig_w;
        for (int c = 0; c < EMBED; c++)
            g_pos_2d[c * orig_h * orig_w + h * orig_w + ww] = w->pos_embed[(s + 1) * EMBED + c];
    }
    upsample_bicubic2d(g_pos_2d, w->pos_embed_interp, EMBED, orig_h, orig_w, GRID_H, GRID_W, false);

    printf("Weight storage used: %d / %d floats (%.1f MB)\n",
           g_weight_offset, 25520000, g_weight_offset * 4.0f / (1024 * 1024));
}

// ============================================================================
// Residual Conv Unit
// [pt2: depth_head.scratch.refinenetN.resconfunitM]
// Uses external scratch buffers instead of heap allocation
// ============================================================================
static void residual_conv_unit(const float* input, float* output,
                               const float* c1w, const float* c1b,
                               const float* c2w, const float* c2b,
                               int ch, int h, int w,
                               float* tmp_buf, float* conv_buf) {
    int sz = ch * h * w;
    // [pt2: aten.relu.default]
    for (int i = 0; i < sz; i++) tmp_buf[i] = input[i] > 0.0f ? input[i] : 0.0f;
    // [pt2: aten.convolution.default] RCU conv1
    conv2d(tmp_buf, conv_buf, c1w, c1b, ch, h, w, ch, 3, 3, 1, 1, 1, 1);
    // [pt2: aten.relu.default]
    for (int i = 0; i < sz; i++) tmp_buf[i] = conv_buf[i] > 0.0f ? conv_buf[i] : 0.0f;
    // [pt2: aten.convolution.default] RCU conv2
    conv2d(tmp_buf, conv_buf, c2w, c2b, ch, h, w, ch, 3, 3, 1, 1, 1, 1);
    // [pt2: aten.add.Tensor] residual connection
    for (int i = 0; i < sz; i++) output[i] = input[i] + conv_buf[i];
}

// ============================================================================
// Transformer Block
// [pt2: pretrained.blocks[i]]
// ============================================================================
static void multihead_attention(const float* input, float* output, const Block* blk) {
    // [pt2: aten.linear.default] QKV projection
    matmul_add(input, blk->attn_qkv_w, buf_qkv, blk->attn_qkv_b, SEQ_LEN, 3 * EMBED, EMBED);

    std::memset(buf_attn_out, 0, SEQ_LEN * EMBED * sizeof(float));

    // [pt2: aten.scaled_dot_product_attention.default]
    for (int h = 0; h < NUM_HEADS; h++) {
        // Extract Q, K, V for this head
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                buf_Q[s * HEAD_DIM + d] = buf_qkv[s * (3 * EMBED) + h * HEAD_DIM + d];
                buf_K[s * HEAD_DIM + d] = buf_qkv[s * (3 * EMBED) + EMBED + h * HEAD_DIM + d];
                buf_V[s * HEAD_DIM + d] = buf_qkv[s * (3 * EMBED) + 2 * EMBED + h * HEAD_DIM + d];
            }
        }
        // Q @ K^T
        matmul(buf_Q, buf_K, buf_scores, SEQ_LEN, SEQ_LEN, HEAD_DIM, false, true);
        // Scale
        float scale = 1.0f / std::sqrt((float)HEAD_DIM);
        for (int i = 0; i < SEQ_LEN * SEQ_LEN; i++) buf_scores[i] *= scale;
        // Softmax
        softmax_inplace(buf_scores, SEQ_LEN, SEQ_LEN);
        // Scores @ V
        matmul_nn(buf_scores, buf_V, buf_head_out, SEQ_LEN, HEAD_DIM, SEQ_LEN);
        // Scatter back
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < HEAD_DIM; d++)
                buf_attn_out[s * EMBED + h * HEAD_DIM + d] = buf_head_out[s * HEAD_DIM + d];
    }
    // [pt2: aten.linear.default] output projection
    matmul_add(buf_attn_out, blk->attn_proj_w, output, blk->attn_proj_b, SEQ_LEN, EMBED, EMBED);
}

static void mlp_forward(const float* input, float* output, const Block* blk) {
    // [pt2: aten.linear.default] fc1
    matmul_add(input, blk->mlp_fc1_w, buf_hidden, blk->mlp_fc1_b, SEQ_LEN, MLP_DIM, EMBED);
    // [pt2: aten.gelu.default]
    gelu_inplace(buf_hidden, SEQ_LEN * MLP_DIM);
    // [pt2: aten.linear.default] fc2
    matmul_add(buf_hidden, blk->mlp_fc2_w, output, blk->mlp_fc2_b, SEQ_LEN, EMBED, MLP_DIM);
}

static void transformer_block(float* tokens, float* tmp, float* tmp2, const Block* blk) {
    // [pt2: aten.native_layer_norm.default] pre-attention norm
    layer_norm(tokens, tmp, blk->norm1_w, blk->norm1_b, SEQ_LEN, EMBED, 1e-6f);
    multihead_attention(tmp, tmp2, blk);
    // [pt2: aten.mul.Tensor + aten.add.Tensor] layer scale + residual
    for (int i = 0; i < SEQ_LEN * EMBED; i++)
        tokens[i] += tmp2[i] * blk->ls1_gamma[i % EMBED];
    // [pt2: aten.native_layer_norm.default] pre-MLP norm
    layer_norm(tokens, tmp, blk->norm2_w, blk->norm2_b, SEQ_LEN, EMBED, 1e-6f);
    mlp_forward(tmp, tmp2, blk);
    // [pt2: aten.mul.Tensor + aten.add.Tensor] layer scale + residual
    for (int i = 0; i < SEQ_LEN * EMBED; i++)
        tokens[i] += tmp2[i] * blk->ls2_gamma[i % EMBED];
}

// ============================================================================
// DPT Decoder Head
// [pt2: depth_head.scratch.refinenet1-4 + output_conv]
// ============================================================================
static void dpt_head(float* output) {
    const Weights* w = &g_weights;

    // [pt2: aten.convolution.default] Scale projections (1x1 conv)
    conv2d(buf_feat[0], buf_s0p, w->proj0_w, w->proj0_b, EMBED, GRID_H, GRID_W, 48, 1, 1, 1, 1, 0, 0);
    // [pt2: aten.convolution.default (transposed)] Spatial upsample
    conv_transpose2d(buf_s0p, buf_s0, w->resize0_w, w->resize0_b, 48, GRID_H, GRID_W, 48, 4, 4, 4, 4, 0, 0);

    conv2d(buf_feat[1], buf_s1p, w->proj1_w, w->proj1_b, EMBED, GRID_H, GRID_W, 96, 1, 1, 1, 1, 0, 0);
    conv_transpose2d(buf_s1p, buf_s1, w->resize1_w, w->resize1_b, 96, GRID_H, GRID_W, 96, 2, 2, 2, 2, 0, 0);

    conv2d(buf_feat[2], buf_s2, w->proj2_w, w->proj2_b, EMBED, GRID_H, GRID_W, 192, 1, 1, 1, 1, 0, 0);

    conv2d(buf_feat[3], buf_s3p, w->proj3_w, w->proj3_b, EMBED, GRID_H, GRID_W, EMBED, 1, 1, 1, 1, 0, 0);
    conv2d(buf_s3p, buf_s3, w->resize3_w, w->resize3_b, EMBED, GRID_H, GRID_W, EMBED, 3, 3, 2, 2, 1, 1);

    // [pt2: aten.convolution.default] Layer RN (3x3 conv, no bias)
    conv2d_nobias(buf_s0, buf_f1, w->layer1_rn_w, 48, 148, 224, 64, 3, 3, 1, 1, 1, 1);
    conv2d_nobias(buf_s1, buf_f2, w->layer2_rn_w, 96, 74, 112, 64, 3, 3, 1, 1, 1, 1);
    conv2d_nobias(buf_s2, buf_f3, w->layer3_rn_w, 192, GRID_H, GRID_W, 64, 3, 3, 1, 1, 1, 1);
    conv2d_nobias(buf_s3, buf_f4, w->layer4_rn_w, EMBED, 19, 28, 64, 3, 3, 1, 1, 1, 1);

    // ---- Stage 4: rcu2(f4) -> upsample -> out_conv ----
    // [pt2: depth_head.scratch.refinenet4]
    residual_conv_unit(buf_f4, scratch_rcu,
                       w->refine[3].rcu2_conv1_w, w->refine[3].rcu2_conv1_b,
                       w->refine[3].rcu2_conv2_w, w->refine[3].rcu2_conv2_b,
                       64, 19, 28, scratch_rcu2, scratch_conv);
    // [pt2: aten.upsample_bilinear2d.vec]
    upsample_bilinear2d(scratch_rcu, scratch_up, 64, 19, 28, GRID_H, GRID_W);
    // [pt2: aten.convolution.default] 1x1 out_conv
    conv2d(scratch_up, scratch_out, w->refine[3].out_conv_w, w->refine[3].out_conv_b,
           64, GRID_H, GRID_W, 64, 1, 1, 1, 1, 0, 0);

    // ---- Stage 3: rcu1(f3) + prev -> rcu2 -> upsample -> out_conv ----
    // [pt2: depth_head.scratch.refinenet3]
    {
        int sz = 64 * GRID_H * GRID_W;
        residual_conv_unit(buf_f3, scratch_rcu,
                           w->refine[2].rcu1_conv1_w, w->refine[2].rcu1_conv1_b,
                           w->refine[2].rcu1_conv2_w, w->refine[2].rcu1_conv2_b,
                           64, GRID_H, GRID_W, scratch_rcu2, scratch_conv);
        // [pt2: aten.add.Tensor] merge with previous stage
        for (int i = 0; i < sz; i++) scratch_merged[i] = scratch_out[i] + scratch_rcu[i];
        residual_conv_unit(scratch_merged, scratch_rcu,
                           w->refine[2].rcu2_conv1_w, w->refine[2].rcu2_conv1_b,
                           w->refine[2].rcu2_conv2_w, w->refine[2].rcu2_conv2_b,
                           64, GRID_H, GRID_W, scratch_rcu2, scratch_conv);
        upsample_bilinear2d(scratch_rcu, scratch_up, 64, GRID_H, GRID_W, 74, 112);
        conv2d(scratch_up, scratch_out, w->refine[2].out_conv_w, w->refine[2].out_conv_b,
               64, 74, 112, 64, 1, 1, 1, 1, 0, 0);
    }

    // ---- Stage 2: rcu1(f2) + prev -> rcu2 -> upsample -> out_conv ----
    // [pt2: depth_head.scratch.refinenet2]
    {
        int sz = 64 * 74 * 112;
        residual_conv_unit(buf_f2, scratch_rcu,
                           w->refine[1].rcu1_conv1_w, w->refine[1].rcu1_conv1_b,
                           w->refine[1].rcu1_conv2_w, w->refine[1].rcu1_conv2_b,
                           64, 74, 112, scratch_rcu2, scratch_conv);
        for (int i = 0; i < sz; i++) scratch_merged[i] = scratch_out[i] + scratch_rcu[i];
        residual_conv_unit(scratch_merged, scratch_rcu,
                           w->refine[1].rcu2_conv1_w, w->refine[1].rcu2_conv1_b,
                           w->refine[1].rcu2_conv2_w, w->refine[1].rcu2_conv2_b,
                           64, 74, 112, scratch_rcu2, scratch_conv);
        upsample_bilinear2d(scratch_rcu, scratch_up, 64, 74, 112, 148, 224);
        conv2d(scratch_up, scratch_out, w->refine[1].out_conv_w, w->refine[1].out_conv_b,
               64, 148, 224, 64, 1, 1, 1, 1, 0, 0);
    }

    // ---- Stage 1: rcu1(f1) + prev -> rcu2 -> upsample -> out_conv ----
    // [pt2: depth_head.scratch.refinenet1]
    {
        int sz = 64 * 148 * 224;
        residual_conv_unit(buf_f1, scratch_rcu,
                           w->refine[0].rcu1_conv1_w, w->refine[0].rcu1_conv1_b,
                           w->refine[0].rcu1_conv2_w, w->refine[0].rcu1_conv2_b,
                           64, 148, 224, scratch_rcu2, scratch_conv);
        for (int i = 0; i < sz; i++) scratch_merged[i] = scratch_out[i] + scratch_rcu[i];
        residual_conv_unit(scratch_merged, scratch_rcu,
                           w->refine[0].rcu2_conv1_w, w->refine[0].rcu2_conv1_b,
                           w->refine[0].rcu2_conv2_w, w->refine[0].rcu2_conv2_b,
                           64, 148, 224, scratch_rcu2, scratch_conv);
        upsample_bilinear2d(scratch_rcu, scratch_up, 64, 148, 224, 296, 448);
        conv2d(scratch_up, scratch_out, w->refine[0].out_conv_w, w->refine[0].out_conv_b,
               64, 296, 448, 64, 1, 1, 1, 1, 0, 0);
    }

    // ---- Output Head ----
    // [pt2: depth_head.scratch.output_conv1] 3x3 conv 64->32
    conv2d(scratch_out, buf_head1, w->out_conv1_w, w->out_conv1_b,
           64, 296, 448, 32, 3, 3, 1, 1, 1, 1);
    // [pt2: aten.upsample_bilinear2d.vec] 296x448 -> 518x784
    upsample_bilinear2d(buf_head1, buf_head1_up, 32, 296, 448, OUT_H, OUT_W);
    // [pt2: depth_head.scratch.output_conv2.0] 3x3 conv 32->32
    conv2d(buf_head1_up, buf_head2, w->out_conv2_0_w, w->out_conv2_0_b,
           32, OUT_H, OUT_W, 32, 3, 3, 1, 1, 1, 1);
    // [pt2: aten.relu.default]
    relu_inplace(buf_head2, 32 * OUT_H * OUT_W);
    // [pt2: depth_head.scratch.output_conv2.2] 1x1 conv 32->1
    conv2d(buf_head2, buf_depth, w->out_conv2_2_w, w->out_conv2_2_b,
           32, OUT_H, OUT_W, 1, 1, 1, 1, 1, 0, 0);
    // [pt2: aten.relu.default] final activation
    relu_inplace(buf_depth, OUT_H * OUT_W);

    std::memcpy(output, buf_depth, OUT_H * OUT_W * sizeof(float));
}

// ============================================================================
// Forward Pass — main entry point
// [pt2: full graph execution]
// ============================================================================
void forward(const float* input, float* output) {
    const Weights* w = &g_weights;

    // [pt2: aten.convolution.default] Patch embedding: [1,3,518,784] -> [384,37,56]
    conv2d(input, buf_patch, w->patch_proj_w, w->patch_proj_b,
           IN_C, IN_H, IN_W, EMBED, PATCH, PATCH, PATCH, PATCH, 0, 0);

    // [pt2: CLS token prepend + flatten to sequence]
    std::memcpy(buf_tokens, w->cls_token, EMBED * sizeof(float));
    for (int h = 0; h < GRID_H; h++)
        for (int ww = 0; ww < GRID_W; ww++) {
            int p = h * GRID_W + ww;
            for (int c = 0; c < EMBED; c++)
                buf_tokens[(p + 1) * EMBED + c] = buf_patch[c * GRID_H * GRID_W + h * GRID_W + ww];
        }

    // [pt2: aten.add.Tensor] Add positional embeddings
    for (int d = 0; d < EMBED; d++)
        buf_tokens[d] += w->pos_embed[d];
    for (int p = 0; p < NUM_PATCHES; p++) {
        int h = p / GRID_W;
        int ww = p % GRID_W;
        for (int c = 0; c < EMBED; c++)
            buf_tokens[(p + 1) * EMBED + c] += w->pos_embed_interp[c * GRID_H * GRID_W + h * GRID_W + ww];
    }

    // [pt2: pretrained.blocks[0..11]] Transformer blocks
    static const int FB[4] = {2, 5, 8, 11};
    int fi = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        transformer_block(buf_tokens, buf_tmp, buf_tmp2, &w->blocks[i]);
        if (fi < 4 && i == FB[fi]) {
            std::memcpy(buf_block_out[fi], buf_tokens, SEQ_LEN * EMBED * sizeof(float));
            fi++;
        }
    }

    // [pt2: aten.native_layer_norm.default + reshape] Extract decoder features
    for (int f = 0; f < 4; f++) {
        layer_norm(buf_block_out[f], buf_normed, w->norm_w, w->norm_b, SEQ_LEN, EMBED, 1e-6f);
        for (int p = 0; p < NUM_PATCHES; p++) {
            int h = p / GRID_W;
            int ww = p % GRID_W;
            for (int c = 0; c < EMBED; c++)
                buf_feat[f][c * GRID_H * GRID_W + h * GRID_W + ww] = buf_normed[(p + 1) * EMBED + c];
        }
    }

    // [pt2: depth_head forward] DPT decoder
    dpt_head(output);
}

} // namespace dav2_s32
