// funasr-encoder: ggml C++ forward pass for the Fun-ASR-Nano audio encoder
// (SenseVoice SAN-M, 50+20 layers) + Transformer adaptor.
//
// Input : fbank.bin (T x 560 f32, the encoder input features)
// Output: out.bin   (T' x 1024 f32, audio embeddings for the LLM)
// Weights: funasr-encoder.gguf (exported by export_encoder_gguf.py)
//
// Validated layer-by-layer against PyTorch golden dumps. fbank is currently
// produced in Python; porting the fbank frontend to C++ is the remaining piece.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct cfg {
    int input_size = 560, d_model = 512, n_head = 4, ffn = 2048;
    int num_blocks = 50, tp_blocks = 20, kernel = 11;
    int adp_llm = 1024, adp_ffn = 2048, adp_layers = 2, adp_head = 8;
};

static const float LN_EPS = 1e-5f;

struct funasr_model {
    cfg c;
    struct ggml_context * ctx_w = nullptr;   // weights (CPU malloc, data set)
    std::map<std::string, struct ggml_tensor *> t;
    struct ggml_tensor * get(const std::string & n) {
        auto it = t.find(n);
        if (it == t.end()) { fprintf(stderr, "missing tensor: %s\n", n.c_str()); exit(1); }
        return it->second;
    }
};

static bool load_model(const char * path, funasr_model & m) {
    struct gguf_init_params p = { /*no_alloc=*/false, /*ctx=*/&m.ctx_w };
    struct gguf_context * gguf = gguf_init_from_file(path, p);
    if (!gguf) { fprintf(stderr, "failed to load gguf %s\n", path); return false; }
    auto rd = [&](const char * k, int def) {
        int i = gguf_find_key(gguf, k); return i < 0 ? def : (int) gguf_get_val_u32(gguf, i);
    };
    m.c.input_size = rd("funasr.enc.input_size", 560);
    m.c.d_model    = rd("funasr.enc.output_size", 512);
    m.c.n_head     = rd("funasr.enc.attention_heads", 4);
    m.c.ffn        = rd("funasr.enc.linear_units", 2048);
    m.c.num_blocks = rd("funasr.enc.num_blocks", 50);
    m.c.tp_blocks  = rd("funasr.enc.tp_blocks", 20);
    m.c.kernel     = rd("funasr.enc.kernel_size", 11);
    m.c.adp_llm    = rd("funasr.adp.llm_dim", 1024);
    m.c.adp_ffn    = rd("funasr.adp.ffn_dim", 2048);
    m.c.adp_layers = rd("funasr.adp.n_layer", 2);
    m.c.adp_head   = rd("funasr.adp.attention_heads", 8);
    int n = gguf_get_n_tensors(gguf);
    for (int i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        m.t[name] = ggml_get_tensor(m.ctx_w, name);
    }
    fprintf(stderr, "loaded %d tensors; cfg: d_model=%d heads=%d blocks=%d tp=%d kernel=%d adp_llm=%d\n",
            n, m.c.d_model, m.c.n_head, m.c.num_blocks, m.c.tp_blocks, m.c.kernel, m.c.adp_llm);
    gguf_free(gguf);
    return true;
}

// helpers ---------------------------------------------------------------
static struct ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);   // [out, T]
    if (b) y = ggml_add(ctx, y, b);
    return y;
}
static struct ggml_tensor * layernorm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * g, ggml_tensor * b) {
    x = ggml_norm(ctx, x, LN_EPS);
    x = ggml_mul(ctx, x, g);
    x = ggml_add(ctx, x, b);
    return x;
}

// SAN-M self-attention + FSMN. x:[D_in,T] -> [d_model,T]
static struct ggml_tensor * sanm_attn(ggml_context * ctx, funasr_model & m, const std::string & pfx,
                                      ggml_tensor * x, int T) {
    const int D = m.c.d_model, H = m.c.n_head, dk = D / H, K = m.c.kernel;
    struct ggml_tensor * qkv = linear(ctx, m.get(pfx + "linear_q_k_v.weight"),
                                      m.get(pfx + "linear_q_k_v.bias"), x);   // [3D, T]
    size_t nb1 = qkv->nb[1];
    struct ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, D, T, nb1, 0));
    struct ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, D, T, nb1, (size_t) D * sizeof(float)));
    struct ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, D, T, nb1, (size_t) 2 * D * sizeof(float)));

    // FSMN: depthwise conv1d along time (per-channel kernel K, "same" padding),
    // plus residual v. Implemented as an exact f32 shift-accumulate to avoid the
    // F16-only ggml_conv_1d_dw path. fsmn kernel stored as [D, K] (ne0=D, ne1=K).
    const int pad = (K - 1) / 2;
    struct ggml_tensor * fk = m.get(pfx + "fsmn_block.weight");               // [D, K]
    struct ggml_tensor * vpad = ggml_pad_ext(ctx, v, 0, 0, pad, pad, 0, 0, 0, 0); // [D, T+2*pad]
    struct ggml_tensor * fsmn = v;                                            // residual
    for (int j = 0; j < K; j++) {
        struct ggml_tensor * sl = ggml_view_2d(ctx, vpad, D, T, vpad->nb[1], (size_t) j * vpad->nb[1]);
        struct ggml_tensor * wj = ggml_view_1d(ctx, fk, D, (size_t) j * fk->nb[1]);
        fsmn = ggml_add(ctx, fsmn, ggml_mul(ctx, ggml_cont(ctx, sl), wj));
    }

    // multi-head attention
    q = ggml_reshape_3d(ctx, q, dk, H, T);
    k = ggml_reshape_3d(ctx, k, dk, H, T);
    struct ggml_tensor * vh = ggml_reshape_3d(ctx, v, dk, H, T);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);   // [dk, T, H]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);   // [dk, T, H]
    vh = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 2, 0, 3)); // [T, dk, H]
    struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);      // [T, T, H]
    kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float) dk));
    kq = ggml_soft_max(ctx, kq);
    struct ggml_tensor * kqv = ggml_mul_mat(ctx, vh, kq);   // [dk, T, H]
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);               // [dk, H, T]
    kqv = ggml_cont_2d(ctx, kqv, D, T);                     // [D, T]
    struct ggml_tensor * att = linear(ctx, m.get(pfx + "linear_out.weight"),
                                      m.get(pfx + "linear_out.bias"), kqv);
    return ggml_add(ctx, att, fsmn);
}

// one SAN-M encoder layer. in_size may differ from d_model (first layer)
static struct ggml_tensor * sanm_layer(ggml_context * ctx, funasr_model & m, const std::string & pfx,
                                       ggml_tensor * x, int T, bool residual_attn) {
    struct ggml_tensor * res = x;
    struct ggml_tensor * h = layernorm(ctx, x, m.get(pfx + "norm1.weight"), m.get(pfx + "norm1.bias"));
    struct ggml_tensor * sa = sanm_attn(ctx, m, pfx + "self_attn.", h, T);
    x = residual_attn ? ggml_add(ctx, res, sa) : sa;
    res = x;
    h = layernorm(ctx, x, m.get(pfx + "norm2.weight"), m.get(pfx + "norm2.bias"));
    h = linear(ctx, m.get(pfx + "feed_forward.w_1.weight"), m.get(pfx + "feed_forward.w_1.bias"), h);
    h = ggml_relu(ctx, h);
    h = linear(ctx, m.get(pfx + "feed_forward.w_2.weight"), m.get(pfx + "feed_forward.w_2.bias"), h);
    return ggml_add(ctx, res, h);
}

// standard transformer layer (adaptor). d=adp_llm
static struct ggml_tensor * adp_layer(ggml_context * ctx, funasr_model & m, const std::string & pfx,
                                      ggml_tensor * x, int T) {
    const int D = m.c.adp_llm, H = m.c.adp_head, dk = D / H;
    struct ggml_tensor * res = x;
    struct ggml_tensor * h = layernorm(ctx, x, m.get(pfx + "norm1.weight"), m.get(pfx + "norm1.bias"));
    struct ggml_tensor * q = linear(ctx, m.get(pfx + "self_attn.linear_q.weight"), m.get(pfx + "self_attn.linear_q.bias"), h);
    struct ggml_tensor * k = linear(ctx, m.get(pfx + "self_attn.linear_k.weight"), m.get(pfx + "self_attn.linear_k.bias"), h);
    struct ggml_tensor * v = linear(ctx, m.get(pfx + "self_attn.linear_v.weight"), m.get(pfx + "self_attn.linear_v.bias"), h);
    q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, dk, H, T), 0, 2, 1, 3);
    k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, dk, H, T), 0, 2, 1, 3);
    struct ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, dk, H, T), 1, 2, 0, 3));
    struct ggml_tensor * kq = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f / sqrtf((float) dk)));
    struct ggml_tensor * kqv = ggml_cont_2d(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vh, kq), 0, 2, 1, 3), D, T);
    struct ggml_tensor * att = linear(ctx, m.get(pfx + "self_attn.linear_out.weight"), m.get(pfx + "self_attn.linear_out.bias"), kqv);
    x = ggml_add(ctx, res, att);
    res = x;
    h = layernorm(ctx, x, m.get(pfx + "norm2.weight"), m.get(pfx + "norm2.bias"));
    h = linear(ctx, m.get(pfx + "feed_forward.w_1.weight"), m.get(pfx + "feed_forward.w_1.bias"), h);
    h = ggml_relu(ctx, h);
    h = linear(ctx, m.get(pfx + "feed_forward.w_2.weight"), m.get(pfx + "feed_forward.w_2.bias"), h);
    return ggml_add(ctx, res, h);
}

// sinusoidal position encoding, depth = input feature dim, positions 1..T
static void add_posenc(std::vector<float> & x, int T, int depth) {
    double inc = log(10000.0) / (depth / 2.0 - 1.0);
    for (int t = 0; t < T; t++) {
        double pos = t + 1;  // positions start at 1
        for (int i = 0; i < depth / 2; i++) {
            double its = exp(i * -inc);
            double st = pos * its;
            x[(size_t) t * depth + i]             += (float) sin(st);
            x[(size_t) t * depth + depth / 2 + i] += (float) cos(st);
        }
    }
}

int main(int argc, char ** argv) {
    std::string gguf_path, fbank_path, out_path = "out.bin";
    int limit = -1;          // -L: run only first N (encoders0+encoders) layers, dump running x
    bool run_adaptor = true;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m") && i+1 < argc) gguf_path  = argv[++i];
        else if (!strcmp(argv[i], "-f") && i+1 < argc) fbank_path = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path   = argv[++i];
        else if (!strcmp(argv[i], "-L") && i+1 < argc) { limit = atoi(argv[++i]); run_adaptor = false; }
        else { fprintf(stderr, "usage: %s -m enc.gguf -f fbank.bin [-o out.bin] [-L nlayers]\n", argv[0]); return 1; }
    }

    funasr_model m;
    if (!load_model(gguf_path.c_str(), m)) return 1;

    // read fbank.bin (T x F)
    FILE * f = fopen(fbank_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", fbank_path.c_str()); return 1; }
    int32_t T, F; if (fread(&T, 4, 1, f) != 1 || fread(&F, 4, 1, f) != 1) { fclose(f); return 1; }
    std::vector<float> fbank((size_t) T * F);
    if (fread(fbank.data(), sizeof(float), fbank.size(), f) != fbank.size()) { fclose(f); return 1; }
    fclose(f);
    fprintf(stderr, "fbank: T=%d F=%d\n", T, F);

    // pre-scale (*sqrt(d_model)) and add position encoding on the host
    float scale = sqrtf((float) m.c.d_model);
    for (auto & v : fbank) v *= scale;
    add_posenc(fbank, T, F);

    // backend + compute context
    ggml_backend_t backend = ggml_backend_cpu_init();
    size_t ctx_size = (size_t) 1024*1024*1024;  // graph metadata
    struct ggml_init_params cp = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(cp);

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, T);
    ggml_set_name(inp, "inp");
    ggml_set_input(inp);

    struct ggml_tensor * x = inp;
    int done = 0;
    bool stop = false;
    // encoders0 (1 layer, in_size=input_size != d_model -> no attn residual)
    x = sanm_layer(ctx, m, "audio_encoder.encoders0.0.", x, T, /*residual_attn=*/false);
    done++;
    if (limit >= 0 && done >= limit) stop = true;
    // encoders (num_blocks-1 layers)
    for (int i = 0; i < m.c.num_blocks - 1 && !stop; i++) {
        x = sanm_layer(ctx, m, "audio_encoder.encoders." + std::to_string(i) + ".", x, T, true);
        done++;
        if (limit >= 0 && done >= limit) stop = true;
    }
    if (!stop) {
        x = layernorm(ctx, x, m.get("audio_encoder.after_norm.weight"), m.get("audio_encoder.after_norm.bias"));
        for (int i = 0; i < m.c.tp_blocks; i++)
            x = sanm_layer(ctx, m, "audio_encoder.tp_encoders." + std::to_string(i) + ".", x, T, true);
        x = layernorm(ctx, x, m.get("audio_encoder.tp_norm.weight"), m.get("audio_encoder.tp_norm.bias"));
        // adaptor: downsample_rate=1 -> linear1(relu)linear2 then blocks
        if (run_adaptor) {
            x = linear(ctx, m.get("audio_adaptor.linear1.weight"), m.get("audio_adaptor.linear1.bias"), x);
            x = ggml_relu(ctx, x);
            x = linear(ctx, m.get("audio_adaptor.linear2.weight"), m.get("audio_adaptor.linear2.bias"), x);
            for (int i = 0; i < m.c.adp_layers; i++)
                x = adp_layer(ctx, m, "audio_adaptor.blocks." + std::to_string(i) + ".", x, T);
        }
    }
    ggml_set_output(x);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_gallocr_alloc_graph(galloc, gf);
    ggml_backend_tensor_set(inp, fbank.data(), 0, ggml_nbytes(inp));

    ggml_backend_cpu_set_n_threads(backend, 8);
    int64_t t0 = ggml_time_us();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }
    int64_t t1 = ggml_time_us();

    int D = (int) x->ne[0];
    std::vector<float> out((size_t) D * T);
    ggml_backend_tensor_get(x, out.data(), 0, ggml_nbytes(x));
    FILE * fo = fopen(out_path.c_str(), "wb");
    if (!fo) { fprintf(stderr, "failed to open output file %s\n", out_path.c_str()); return 1; }
    fwrite(&T, 4, 1, fo); fwrite(&D, 4, 1, fo); fwrite(out.data(), sizeof(float), out.size(), fo);
    fclose(fo);
    fprintf(stderr, "done: wrote %s [%d x %d] in %.2f s (layers run=%d, adaptor=%d)\n",
            out_path.c_str(), T, D, (t1 - t0)/1e6, done, run_adaptor && !stop);
    ggml_gallocr_free(galloc); ggml_free(ctx); ggml_backend_free(backend);
    if (m.ctx_w) ggml_free(m.ctx_w);
    return 0;
}
