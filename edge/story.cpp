// End-to-end proof: a real trained transformer generating text on-device,
// with every attention op running through this repo's CPU kernel.
//
// The model is TinyStories (Karpathy's Llama-architecture stories42M:
// dim 512, 8 layers, 8 heads, head_dim 64, vocab 32000), a small LM that
// writes coherent children's stories. The point is not the model, it is
// that the whole forward pass runs on a CPU with no GPU in sight: the
// prompt prefill goes through attention_flash_fast (NEON + threaded), and
// each generated token attends over the KV cache through attention_step_cpu.
// Everything else (RMSNorm, RoPE, SwiGLU, the matmuls, the tokenizer) is
// here so the binary is self-contained and runs the same on an Orange Pi.
//
// Checkpoint/tokenizer format follow Karpathy's llama2.c so the public
// weights load directly. Greedy sampling (--temp 0) is deterministic and
// lets the output be checked token-for-token against the llama2.c
// reference; a temperature is available for nicer stories.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "attention.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// ---- config + weights (llama2.c legacy checkpoint layout) ----

struct Config {
    int dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq_len;
};

struct Weights {
    const float *tok_emb, *rms_att, *wq, *wk, *wv, *wo, *rms_ffn;
    const float *w1, *w2, *w3, *rms_final, *freq_real, *freq_imag, *wcls;
};

static const float *map_checkpoint(const char *path, Config &c, Weights &w,
                                   size_t &map_len)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open checkpoint"); exit(1); }
    struct stat st;
    fstat(fd, &st);
    map_len = st.st_size;
    void *m = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }

    const int *hdr = (const int *)m;
    c = { hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6] };
    int shared = c.vocab > 0;
    if (c.vocab < 0) c.vocab = -c.vocab;

    const int hd = c.dim / c.n_heads;
    const int kv = c.n_kv_heads * hd;
    const float *p = (const float *)(hdr + 7);
    w.tok_emb = p;  p += (size_t)c.vocab * c.dim;
    w.rms_att = p;  p += (size_t)c.n_layers * c.dim;
    w.wq = p;       p += (size_t)c.n_layers * c.dim * (c.n_heads * hd);
    w.wk = p;       p += (size_t)c.n_layers * c.dim * kv;
    w.wv = p;       p += (size_t)c.n_layers * c.dim * kv;
    w.wo = p;       p += (size_t)c.n_layers * (c.n_heads * hd) * c.dim;
    w.rms_ffn = p;  p += (size_t)c.n_layers * c.dim;
    w.w1 = p;       p += (size_t)c.n_layers * c.hidden * c.dim;
    w.w2 = p;       p += (size_t)c.n_layers * c.dim * c.hidden;
    w.w3 = p;       p += (size_t)c.n_layers * c.hidden * c.dim;
    w.rms_final = p; p += c.dim;
    w.freq_real = p; p += (size_t)c.seq_len * hd / 2;
    w.freq_imag = p; p += (size_t)c.seq_len * hd / 2;
    w.wcls = shared ? w.tok_emb : p;
    return (const float *)m;
}

// ---- math kernels (NEON where available) ----

// out[i] = sum_j W[i*n + j] * x[j], W is [d][n] row-major
static void matvec(float *out, const float *x, const float *W, int n, int d)
{
    for (int i = 0; i < d; i++) {
        const float *row = W + (size_t)i * n;
#if defined(__ARM_NEON)
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            a0 = vfmaq_f32(a0, vld1q_f32(row + j),     vld1q_f32(x + j));
            a1 = vfmaq_f32(a1, vld1q_f32(row + j + 4), vld1q_f32(x + j + 4));
        }
        float s = vaddvq_f32(vaddq_f32(a0, a1));
        for (; j < n; j++) s += row[j] * x[j];
        out[i] = s;
#else
        float s = 0;
        for (int j = 0; j < n; j++) s += row[j] * x[j];
        out[i] = s;
#endif
    }
}

static void rmsnorm(float *out, const float *x, const float *w, int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

// RoPE on q (dim) and k (kv_dim) for position pos, using the checkpoint's
// precomputed cos/sin tables.
static void rope(float *q, float *k, int pos, const Config &c,
                 const Weights &w, int kv_dim)
{
    const int hd = c.dim / c.n_heads;
    const float *fr = w.freq_real + (size_t)pos * hd / 2;
    const float *fi = w.freq_imag + (size_t)pos * hd / 2;
    for (int j = 0; j < c.dim; j += 2) {
        const int local = j % hd;           // pair index within a head
        const float fcr = fr[local / 2], fci = fi[local / 2];
        float q0 = q[j], q1 = q[j + 1];
        q[j] = q0 * fcr - q1 * fci;
        q[j + 1] = q0 * fci + q1 * fcr;
        if (j < kv_dim) {
            float k0 = k[j], k1 = k[j + 1];
            k[j] = k0 * fcr - k1 * fci;
            k[j + 1] = k0 * fci + k1 * fcr;
        }
    }
}

static void softmax(float *x, int n)
{
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - m); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ---- tokenizer (llama2.c format) ----

struct Tokenizer {
    std::vector<std::string> vocab;
    std::vector<float> score;
    std::unordered_map<std::string, int> lookup;

    void load(const char *path, int vocab_size)
    {
        FILE *f = fopen(path, "rb");
        if (!f) { perror("open tokenizer"); exit(1); }
        int max_len;
        if (fread(&max_len, sizeof(int), 1, f) != 1) { exit(1); }
        vocab.resize(vocab_size);
        score.resize(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            float sc; int len;
            if (fread(&sc, 4, 1, f) != 1 || fread(&len, 4, 1, f) != 1) exit(1);
            std::string s(len, '\0');
            if (len && fread(&s[0], 1, len, f) != (size_t)len) exit(1);
            vocab[i] = s; score[i] = sc; lookup[s] = i;
        }
        fclose(f);
    }

    int id(const std::string &s) const
    {
        auto it = lookup.find(s);
        return it == lookup.end() ? -1 : it->second;
    }

    // BPE encode with a dummy leading space and byte fallback, then greedy
    // score-maximizing merges (matches llama2.c bpe_encode).
    std::vector<int> encode(const std::string &text, bool bos) const
    {
        std::vector<int> toks;
        if (bos) toks.push_back(1);
        if (!text.empty()) {
            int sp = id(" ");
            if (sp >= 0) toks.push_back(sp);
        }
        for (size_t i = 0; i < text.size();) {
            // consume one UTF-8 codepoint
            size_t len = 1;
            unsigned char ch = text[i];
            if (ch >= 0xF0) len = 4; else if (ch >= 0xE0) len = 3;
            else if (ch >= 0xC0) len = 2;
            std::string cp = text.substr(i, len);
            int t = id(cp);
            if (t >= 0) toks.push_back(t);
            else for (unsigned char b : cp) toks.push_back((int)b + 3);  // byte tokens
            i += len;
        }
        for (;;) {
            float best = -1e10f; int best_id = -1; size_t best_at = 0;
            for (size_t i = 0; i + 1 < toks.size(); i++) {
                int m = id(vocab[toks[i]] + vocab[toks[i + 1]]);
                if (m >= 0 && score[m] > best) { best = score[m]; best_id = m; best_at = i; }
            }
            if (best_id < 0) break;
            toks[best_at] = best_id;
            toks.erase(toks.begin() + best_at + 1);
        }
        return toks;
    }

    // Print one token (llama2.c decode rules: strip the space after BOS,
    // expand raw <0xNN> byte tokens).
    void print(int prev, int tok) const
    {
        const char *p = vocab[tok].c_str();
        if (prev == 1 && p[0] == ' ') p++;
        unsigned char byte;
        if (std::sscanf(p, "<0x%02hhX>", &byte) == 1) {
            std::fputc(byte, stdout);
        } else {
            std::fputs(p, stdout);
        }
    }
};

// ---- run state ----

struct State {
    std::vector<float> x, xb, xb2, hb, hb2, q, k, v, att, logits;
    std::vector<float> key_cache, value_cache;      // [layer][pos][kv_dim]
    void init(const Config &c, int kv_dim)
    {
        x.assign(c.dim, 0); xb.assign(c.dim, 0); xb2.assign(c.dim, 0);
        hb.assign(c.hidden, 0); hb2.assign(c.hidden, 0);
        q.assign(c.dim, 0); k.assign(kv_dim, 0); v.assign(kv_dim, 0);
        att.assign(c.dim, 0); logits.assign(c.vocab, 0);
        key_cache.assign((size_t)c.n_layers * c.seq_len * kv_dim, 0);
        value_cache.assign((size_t)c.n_layers * c.seq_len * kv_dim, 0);
    }
};

static uint64_t rng_state = 0x2545F4914F6CDD1DULL;
static float rng_float()
{
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)((rng_state * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

static int sample(float *logits, int n, float temp)
{
    if (temp <= 0.0f) {                         // greedy: deterministic
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);
    float r = rng_float(), cdf = 0;
    for (int i = 0; i < n; i++) { cdf += logits[i]; if (r < cdf) return i; }
    return n - 1;
}

// One transformer layer's FFN + residual, shared by prefill and decode.
static void ffn(State &st, float *x, const Config &c, const Weights &w, int l)
{
    rmsnorm(st.xb.data(), x, w.rms_ffn + (size_t)l * c.dim, c.dim);
    matvec(st.hb.data(), st.xb.data(), w.w1 + (size_t)l * c.hidden * c.dim,
           c.dim, c.hidden);
    matvec(st.hb2.data(), st.xb.data(), w.w3 + (size_t)l * c.hidden * c.dim,
           c.dim, c.hidden);
    for (int i = 0; i < c.hidden; i++)
        st.hb[i] = st.hb[i] / (1.0f + std::exp(-st.hb[i])) * st.hb2[i];  // SwiGLU
    matvec(st.xb.data(), st.hb.data(), w.w2 + (size_t)l * c.dim * c.hidden,
           c.hidden, c.dim);
    for (int i = 0; i < c.dim; i++) x[i] += st.xb[i];
}

// Prefill the prompt: attention for all P positions runs through
// attention_flash_fast (the batched, threaded kernel). Fills the KV cache
// so decode can continue. Returns logits for the last position.
static void prefill(State &st, const std::vector<int> &toks, const Config &c,
                    const Weights &w, int kv_dim)
{
    const int P = (int)toks.size();
    const int hd = c.dim / c.n_heads;
    std::vector<float> xall((size_t)P * c.dim);
    for (int p = 0; p < P; p++)
        std::memcpy(&xall[(size_t)p * c.dim], w.tok_emb + (size_t)toks[p] * c.dim,
                    c.dim * sizeof(float));

    // head-major scratch for the attention kernel: [head][pos][hd]
    std::vector<float> Qb((size_t)c.n_heads * P * hd);
    std::vector<float> Kb((size_t)c.n_heads * P * hd);
    std::vector<float> Vb((size_t)c.n_heads * P * hd);
    std::vector<float> Ob((size_t)c.n_heads * P * hd);
    Shape shp{ 1, c.n_heads, P, hd };

    for (int l = 0; l < c.n_layers; l++) {
        float *kc = st.key_cache.data() + (size_t)l * c.seq_len * kv_dim;
        float *vc = st.value_cache.data() + (size_t)l * c.seq_len * kv_dim;
        for (int p = 0; p < P; p++) {
            float *x = &xall[(size_t)p * c.dim];
            rmsnorm(st.xb.data(), x, w.rms_att + (size_t)l * c.dim, c.dim);
            matvec(st.q.data(), st.xb.data(),
                   w.wq + (size_t)l * c.dim * c.dim, c.dim, c.dim);
            matvec(st.k.data(), st.xb.data(),
                   w.wk + (size_t)l * c.dim * kv_dim, c.dim, kv_dim);
            matvec(st.v.data(), st.xb.data(),
                   w.wv + (size_t)l * c.dim * kv_dim, c.dim, kv_dim);
            rope(st.q.data(), st.k.data(), p, c, w, kv_dim);
            std::memcpy(kc + (size_t)p * kv_dim, st.k.data(), kv_dim * sizeof(float));
            std::memcpy(vc + (size_t)p * kv_dim, st.v.data(), kv_dim * sizeof(float));
            for (int h = 0; h < c.n_heads; h++)
                for (int d = 0; d < hd; d++) {
                    Qb[((size_t)h * P + p) * hd + d] = st.q[h * hd + d];
                    Kb[((size_t)h * P + p) * hd + d] = st.k[h * hd + d];
                    Vb[((size_t)h * P + p) * hd + d] = st.v[h * hd + d];
                }
        }
        attention_flash_fast(shp, Qb, Kb, Vb, /*causal=*/true, Ob, nullptr);
        for (int p = 0; p < P; p++) {
            float *x = &xall[(size_t)p * c.dim];
            for (int h = 0; h < c.n_heads; h++)
                for (int d = 0; d < hd; d++)
                    st.att[h * hd + d] = Ob[((size_t)h * P + p) * hd + d];
            matvec(st.xb2.data(), st.att.data(),
                   w.wo + (size_t)l * c.dim * c.dim, c.dim, c.dim);
            for (int i = 0; i < c.dim; i++) x[i] += st.xb2[i];
            ffn(st, x, c, w, l);
        }
    }
    float *last = &xall[(size_t)(P - 1) * c.dim];
    rmsnorm(st.x.data(), last, w.rms_final, c.dim);
    matvec(st.logits.data(), st.x.data(), w.wcls, c.dim, c.vocab);
}

// One decode step at position pos: attention runs through attention_step_cpu
// against the KV cache. Returns logits.
static void decode(State &st, int token, int pos, const Config &c,
                   const Weights &w, int kv_dim)
{
    const int hd = c.dim / c.n_heads;
    std::memcpy(st.x.data(), w.tok_emb + (size_t)token * c.dim,
                c.dim * sizeof(float));
    for (int l = 0; l < c.n_layers; l++) {
        float *kc = st.key_cache.data() + (size_t)l * c.seq_len * kv_dim;
        float *vc = st.value_cache.data() + (size_t)l * c.seq_len * kv_dim;
        rmsnorm(st.xb.data(), st.x.data(), w.rms_att + (size_t)l * c.dim, c.dim);
        matvec(st.q.data(), st.xb.data(), w.wq + (size_t)l * c.dim * c.dim,
               c.dim, c.dim);
        matvec(st.k.data(), st.xb.data(), w.wk + (size_t)l * c.dim * kv_dim,
               c.dim, kv_dim);
        matvec(st.v.data(), st.xb.data(), w.wv + (size_t)l * c.dim * kv_dim,
               c.dim, kv_dim);
        rope(st.q.data(), st.k.data(), pos, c, w, kv_dim);
        std::memcpy(kc + (size_t)pos * kv_dim, st.k.data(), kv_dim * sizeof(float));
        std::memcpy(vc + (size_t)pos * kv_dim, st.v.data(), kv_dim * sizeof(float));
        attention_step_cpu(st.q.data(), kc, vc, c.n_heads, hd, pos + 1,
                           st.att.data());
        matvec(st.xb2.data(), st.att.data(), w.wo + (size_t)l * c.dim * c.dim,
               c.dim, c.dim);
        for (int i = 0; i < c.dim; i++) st.x[i] += st.xb2[i];
        ffn(st, st.x.data(), c, w, l);
    }
    rmsnorm(st.x.data(), st.x.data(), w.rms_final, c.dim);
    matvec(st.logits.data(), st.x.data(), w.wcls, c.dim, c.vocab);
}

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

int main(int argc, char **argv)
{
    const char *ckpt = "edge/stories42M.bin";
    const char *tok_path = "edge/tokenizer.bin";
    std::string prompt = "Once upon a time";
    int steps = 128;
    float temp = 0.0f;                          // greedy by default
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "-n") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) rng_state = strtoull(argv[++i], 0, 10) | 1;
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) ckpt = argv[++i];
    }

    Config c; Weights w; size_t map_len;
    const float *base = map_checkpoint(ckpt, c, w, map_len);
    const int hd = c.dim / c.n_heads;
    const int kv_dim = c.n_kv_heads * hd;
    Tokenizer tk; tk.load(tok_path, c.vocab);

    std::fprintf(stderr,
        "model: dim=%d layers=%d heads=%d head_dim=%d vocab=%d | prompt=%d tok, "
        "temp=%.2f\n", c.dim, c.n_layers, c.n_heads, hd, c.vocab, 0, temp);

    State st; st.init(c, kv_dim);
    std::vector<int> prompt_toks = tk.encode(prompt, /*bos=*/true);
    const int P = (int)prompt_toks.size();

    // ---- prefill (attention_flash_fast) ----
    auto t_pf = clk::now();
    prefill(st, prompt_toks, c, w, kv_dim);
    double pf_ms = ms_since(t_pf);

    // print the prompt (skip the BOS token, like the reference)
    int prev = -1;
    for (int i = 0; i < P; i++) {
        if (prompt_toks[i] != 1) tk.print(prev, prompt_toks[i]);
        prev = prompt_toks[i];
    }
    std::fflush(stdout);

    // ---- decode (attention_step_cpu) ----
    int token = sample(st.logits.data(), c.vocab, temp);
    int pos = P;
    double dec_ms = 0;
    int generated = 0;
    while (pos < c.seq_len && generated < steps) {
        if (token == 1) break;                  // BOS delimits end of a story
        tk.print(prev, token); std::fflush(stdout);
        prev = token;
        auto t = clk::now();
        decode(st, token, pos, c, w, kv_dim);
        dec_ms += ms_since(t);
        token = sample(st.logits.data(), c.vocab, temp);
        pos++; generated++;
    }
    std::printf("\n");

    std::fprintf(stderr,
        "\n--- on-device timing (this machine) ---\n"
        "prefill: %d prompt tokens in %.1f ms (%.1f tok/s) via attention_flash_fast\n"
        "decode : %d tokens in %.1f ms (%.1f tok/s) via attention_step_cpu\n",
        P, pf_ms, P * 1000.0 / pf_ms,
        generated, dec_ms, generated * 1000.0 / dec_ms);

    munmap((void *)base, map_len);
    return 0;
}
