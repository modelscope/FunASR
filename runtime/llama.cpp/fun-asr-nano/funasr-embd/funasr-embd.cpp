// funasr-embd: decode Fun-ASR-Nano audio embeddings through a Qwen3 GGUF.
//
// Reads an inputs_embeds matrix (produced by the FunASR audio encoder+adaptor,
// concatenated with the text prompt embeddings) and feeds it directly to the
// LLM via llama_decode's embedding input path -- the same mechanism llava/mtmd
// use to inject vision embeddings. This bridges FunASR's audio frontend to the
// llama.cpp / GGUF ecosystem.
//
// embeds.bin format: int32 n_tokens, int32 n_embd, then n_tokens*n_embd float32
// (row-major). n_embd must equal the model's input embedding dim.

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(char ** argv) {
    printf("\nusage: %s -m model.gguf -e embeds.bin [-n n_predict] [-ngl n_gpu_layers]\n\n", argv[0]);
}

// read embeds.bin -> (n_tokens, n_embd, data)
static bool read_embeds(const std::string & path, int & n_tokens, int & n_embd, std::vector<float> & data) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "error: cannot open %s\n", path.c_str()); return false; }
    int32_t hdr[2];
    if (fread(hdr, sizeof(int32_t), 2, f) != 2) { fclose(f); return false; }
    n_tokens = hdr[0];
    n_embd   = hdr[1];
    if (n_tokens <= 0 || n_embd <= 0) { fclose(f); return false; }
    data.resize((size_t) n_tokens * n_embd);
    size_t got = fread(data.data(), sizeof(float), data.size(), f);
    fclose(f);
    if (got != data.size()) {
        fprintf(stderr, "error: short read (%zu/%zu floats)\n", got, data.size());
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, embeds_path;
    int n_predict = 512;
    int ngl = 0; // CPU by default; the whole point is CPU/edge

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")   && i + 1 < argc) model_path  = argv[++i];
        else if (!strcmp(argv[i], "-e")   && i + 1 < argc) embeds_path = argv[++i];
        else if (!strcmp(argv[i], "-n")   && i + 1 < argc) n_predict   = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl") && i + 1 < argc) ngl         = std::stoi(argv[++i]);
        else { print_usage(argv); return 1; }
    }
    if (model_path.empty() || embeds_path.empty()) { print_usage(argv); return 1; }

    int n_tokens = 0, n_embd_in = 0;
    std::vector<float> embd;
    if (!read_embeds(embeds_path, n_tokens, n_embd_in, embd)) return 1;
    fprintf(stderr, "loaded embeds: n_tokens=%d n_embd=%d\n", n_tokens, n_embd_in);

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "error: unable to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd_model = llama_model_n_embd_inp(model);
    if (n_embd_in != n_embd_model) {
        fprintf(stderr, "error: embd dim %d != model input embd dim %d\n", n_embd_in, n_embd_model);
        llama_model_free(model);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = n_tokens + n_predict + 8;
    cparams.n_batch = n_tokens + 8;   // process the whole embd prompt in one ubatch
    cparams.n_ubatch = n_tokens + 8;
    cparams.no_perf = false;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "error: failed to create context\n"); llama_model_free(model); return 1; }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // --- decode the embedding prompt (causal, single sequence, positions 0..n-1) ---
    std::vector<llama_pos>      pos(n_tokens);
    std::vector<int32_t>        n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id>   seq_id_0(1, 0);
    std::vector<llama_seq_id *> seq_id(n_tokens);
    std::vector<int8_t>         logits(n_tokens, 0);
    for (int i = 0; i < n_tokens; i++) { pos[i] = i; seq_id[i] = seq_id_0.data(); }
    logits[n_tokens - 1] = 1; // only need logits for the last position

    llama_batch batch = {
        /*n_tokens =*/ n_tokens,
        /*token    =*/ nullptr,
        /*embd     =*/ embd.data(),
        /*pos      =*/ pos.data(),
        /*n_seq_id =*/ n_seq_id.data(),
        /*seq_id   =*/ seq_id.data(),
        /*logits   =*/ logits.data(),
    };

    const int64_t t_start = ggml_time_us();
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: llama_decode failed on embd prompt\n");
        return 1;
    }

    // --- generation loop ---
    std::string out;
    int n_decode = 0;
    llama_token tok = llama_sampler_sample(smpl, ctx, -1);
    for (int n_pos = n_tokens; n_pos < n_tokens + n_predict; ) {
        if (llama_vocab_is_eog(vocab, tok)) break;
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0) { out.append(buf, n); }
        printf("%.*s", n > 0 ? n : 0, buf);
        fflush(stdout);

        llama_batch tb = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, tb) != 0) { fprintf(stderr, "error: decode failed\n"); return 1; }
        n_pos += 1;
        n_decode += 1;
        tok = llama_sampler_sample(smpl, ctx, -1);
    }
    printf("\n");
    const int64_t t_end = ggml_time_us();
    fprintf(stderr, "\n[funasr-embd] generated %d tokens in %.2f s (%.1f tok/s)\n",
            n_decode, (t_end - t_start) / 1e6, n_decode / ((t_end - t_start) / 1e6));

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
