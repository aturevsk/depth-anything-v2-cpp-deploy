// Debug harness: compares C++ intermediates against PyTorch checkpoints
// to identify exactly where the accuracy diverges.
#include "depth_anything_v2.h"
#include <cstdio>
#include <cmath>

std::vector<float> load_bin(const std::string& path, int n) {
    std::vector<float> data(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return data; }
    f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
    return data;
}

void compare(const char* name, const float* cpp, const float* ref, int n) {
    double max_err = 0, sum_sq = 0, sum_ref_sq = 0;
    double cpp_mean = 0, ref_mean = 0;
    for (int i = 0; i < n; i++) {
        double e = std::abs((double)cpp[i] - (double)ref[i]);
        max_err = std::max(max_err, e);
        sum_sq += e * e;
        cpp_mean += cpp[i];
        ref_mean += ref[i];
        sum_ref_sq += (double)ref[i] * ref[i];
    }
    double rmse = std::sqrt(sum_sq / n);
    double ref_rms = std::sqrt(sum_ref_sq / n);
    cpp_mean /= n; ref_mean /= n;
    printf("%-30s  max_err=%.4e  rmse=%.4e  rel_rmse=%.4e  cpp_mean=%.6f  ref_mean=%.6f\n",
           name, max_err, rmse, ref_rms > 0 ? rmse/ref_rms : 0, cpp_mean, ref_mean);
}

int main() {
    using namespace dav2;
    std::string wdir = "../weights";
    std::string ddir = "../debug";

    printf("=== DEBUG: Checkpoint comparison ===\n\n");

    // Load model
    DepthAnythingV2 model;
    model.load_weights(wdir);
    auto& w = model.w;

    // Load input
    auto input = load_bin("../reference/input.bin", 1*3*518*784);

    // ====== STEP 1: Patch embedding ======
    std::vector<float> patch_embed(EMBED * GRID_H * GRID_W);
    conv2d(input.data(), patch_embed.data(), w.patch_proj_w.data(), w.patch_proj_b.data(),
           IN_C, IN_H, IN_W, EMBED, PATCH, PATCH, PATCH, PATCH, 0, 0);

    auto ref_patches = load_bin(ddir + "/01_patches.bin", 1*EMBED*GRID_H*GRID_W);
    compare("1. Patch embedding", patch_embed.data(), ref_patches.data(), EMBED*GRID_H*GRID_W);

    // ====== STEP 2: Tokens + CLS ======
    std::vector<float> tokens(SEQ_LEN * EMBED);
    std::memcpy(tokens.data(), w.cls_token.data(), EMBED * sizeof(float));
    for (int h = 0; h < GRID_H; h++)
        for (int ww = 0; ww < GRID_W; ww++) {
            int p = h * GRID_W + ww;
            for (int c = 0; c < EMBED; c++)
                tokens[(p+1)*EMBED + c] = patch_embed[c*GRID_H*GRID_W + h*GRID_W + ww];
        }

    auto ref_tokens = load_bin(ddir + "/02_tokens_before_pos.bin", 1*SEQ_LEN*EMBED);
    compare("2. Tokens before pos", tokens.data(), ref_tokens.data(), SEQ_LEN*EMBED);

    // ====== STEP 3: Positional embedding ======
    // CLS pos embed
    for (int d = 0; d < EMBED; d++)
        tokens[d] += w.pos_embed[d];

    // Patch pos embeds: interpolate [37,37] -> [37,56]
    int orig_h = 37, orig_w = 37;
    std::vector<float> pos_2d(EMBED * orig_h * orig_w);
    for (int s = 0; s < orig_h * orig_w; s++) {
        int h = s / orig_w, ww = s % orig_w;
        for (int c = 0; c < EMBED; c++)
            pos_2d[c*orig_h*orig_w + h*orig_w + ww] = w.pos_embed[(s+1)*EMBED + c];
    }
    std::vector<float> pos_interp(EMBED * GRID_H * GRID_W);
    upsample_bicubic2d(pos_2d.data(), pos_interp.data(), EMBED, orig_h, orig_w, GRID_H, GRID_W, false);

    for (int p = 0; p < NUM_PATCHES; p++) {
        int h = p / GRID_W, ww = p % GRID_W;
        for (int c = 0; c < EMBED; c++)
            tokens[(p+1)*EMBED + c] += pos_interp[c*GRID_H*GRID_W + h*GRID_W + ww];
    }

    auto ref_after_pos = load_bin(ddir + "/04_tokens_after_pos.bin", 1*SEQ_LEN*EMBED);
    compare("3. Tokens after pos embed", tokens.data(), ref_after_pos.data(), SEQ_LEN*EMBED);

    // ====== STEP 4: Transformer blocks ======
    std::vector<float> tmp(SEQ_LEN * EMBED), tmp2(SEQ_LEN * EMBED);
    constexpr int FEAT_BLOCKS[4] = {2, 5, 8, 11};
    std::vector<float> block_outputs[4];
    int feat_idx = 0;

    for (int i = 0; i < NUM_BLOCKS; i++) {
        // Pre-norm attention
        layer_norm(tokens.data(), tmp.data(), w.blocks[i].norm1_w.data(),
                   w.blocks[i].norm1_b.data(), SEQ_LEN, EMBED);

        // Multi-head attention (inline to debug)
        {
            const auto& blk = w.blocks[i];
            std::vector<float> qkv(SEQ_LEN * 3 * EMBED);
            matmul_add(tmp.data(), blk.attn_qkv_w.data(), qkv.data(),
                       blk.attn_qkv_b.data(), SEQ_LEN, 3*EMBED, EMBED);

            std::vector<float> attn_out(SEQ_LEN * EMBED, 0.0f);
            for (int h = 0; h < NUM_HEADS; h++) {
                std::vector<float> Q(SEQ_LEN*HEAD_DIM), K(SEQ_LEN*HEAD_DIM), V(SEQ_LEN*HEAD_DIM);
                for (int s = 0; s < SEQ_LEN; s++)
                    for (int d = 0; d < HEAD_DIM; d++) {
                        Q[s*HEAD_DIM+d] = qkv[s*(3*EMBED) + 0*EMBED + h*HEAD_DIM + d];
                        K[s*HEAD_DIM+d] = qkv[s*(3*EMBED) + 1*EMBED + h*HEAD_DIM + d];
                        V[s*HEAD_DIM+d] = qkv[s*(3*EMBED) + 2*EMBED + h*HEAD_DIM + d];
                    }

                std::vector<float> scores(SEQ_LEN*SEQ_LEN);
                matmul(Q.data(), K.data(), scores.data(), SEQ_LEN, SEQ_LEN, HEAD_DIM, false, true);
                float scale = 1.0f / std::sqrt((float)HEAD_DIM);
                for (auto& v : scores) v *= scale;
                softmax_inplace(scores.data(), SEQ_LEN, SEQ_LEN);

                std::vector<float> head_out(SEQ_LEN*HEAD_DIM);
                matmul(scores.data(), V.data(), head_out.data(), SEQ_LEN, HEAD_DIM, SEQ_LEN);

                for (int s = 0; s < SEQ_LEN; s++)
                    for (int d = 0; d < HEAD_DIM; d++)
                        attn_out[s*EMBED + h*HEAD_DIM + d] = head_out[s*HEAD_DIM + d];
            }

            matmul_add(attn_out.data(), blk.attn_proj_w.data(), tmp2.data(),
                       blk.attn_proj_b.data(), SEQ_LEN, EMBED, EMBED);
        }

        // Residual + LayerScale
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < EMBED; d++)
                tokens[s*EMBED+d] += tmp2[s*EMBED+d] * w.blocks[i].ls1_gamma[d];

        // Pre-norm MLP
        layer_norm(tokens.data(), tmp.data(), w.blocks[i].norm2_w.data(),
                   w.blocks[i].norm2_b.data(), SEQ_LEN, EMBED);

        // MLP
        {
            const auto& blk = w.blocks[i];
            std::vector<float> hidden(SEQ_LEN * MLP_DIM);
            matmul_add(tmp.data(), blk.mlp_fc1_w.data(), hidden.data(),
                       blk.mlp_fc1_b.data(), SEQ_LEN, MLP_DIM, EMBED);
            gelu_inplace(hidden.data(), SEQ_LEN * MLP_DIM);
            matmul_add(hidden.data(), blk.mlp_fc2_w.data(), tmp2.data(),
                       blk.mlp_fc2_b.data(), SEQ_LEN, EMBED, MLP_DIM);
        }

        // Residual + LayerScale
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < EMBED; d++)
                tokens[s*EMBED+d] += tmp2[s*EMBED+d] * w.blocks[i].ls2_gamma[d];

        if (feat_idx < 4 && i == FEAT_BLOCKS[feat_idx]) {
            block_outputs[feat_idx].assign(tokens.begin(), tokens.end());

            char fname[128];
            snprintf(fname, sizeof(fname), "%s/05_block%d_output.bin", ddir.c_str(), i);
            auto ref = load_bin(fname, SEQ_LEN*EMBED);
            char label[64];
            snprintf(label, sizeof(label), "4. Block %d output", i);
            compare(label, tokens.data(), ref.data(), SEQ_LEN*EMBED);
            feat_idx++;
        }
    }

    // ====== STEP 5: Feature extraction (norm + reshape) ======
    for (int f = 0; f < 4; f++) {
        std::vector<float> normed(SEQ_LEN * EMBED);
        layer_norm(block_outputs[f].data(), normed.data(),
                   w.norm_w.data(), w.norm_b.data(), SEQ_LEN, EMBED);

        std::vector<float> feat_3d(EMBED * GRID_H * GRID_W);
        for (int p = 0; p < NUM_PATCHES; p++) {
            int h = p / GRID_W, ww = p % GRID_W;
            for (int c = 0; c < EMBED; c++)
                feat_3d[c*GRID_H*GRID_W + h*GRID_W + ww] = normed[(p+1)*EMBED + c];
        }

        char fname[128];
        snprintf(fname, sizeof(fname), "%s/06_feat3d_block%d.bin", ddir.c_str(), FEAT_BLOCKS[f]);
        auto ref = load_bin(fname, EMBED*GRID_H*GRID_W);
        char label[64];
        snprintf(label, sizeof(label), "5. Feat3d block %d", FEAT_BLOCKS[f]);
        compare(label, feat_3d.data(), ref.data(), EMBED*GRID_H*GRID_W);
    }

    printf("\nDone. Fix errors above before proceeding to full forward pass.\n");
    return 0;
}
