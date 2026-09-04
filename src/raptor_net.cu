// RAPTOR foundation policy: Dense(obs->H)+ReLU -> GRU(H) -> actor Dense(H->A).
// The recurrence runs time-major; training buffers arrive as (B, TT, *) and are
// transposed on entry and exit. Included by ocean.cu.

static constexpr int RN_GATES = 3;
static constexpr float RN_LOGSTD_INIT = -5.5f;

// RAPTOR_POLICY_BLOB points at the 2084-float canonical actor blob; unset trains from scratch.
static const float* rn_blob() {
    static std::vector<float> blob;
    static bool tried = false;
    if (!tried) {
        tried = true;
        const char* path = getenv("RAPTOR_POLICY_BLOB");
        if (path) {
            FILE* f = fopen(path, "rb");
            if (!f) {
                fprintf(stderr, "RAPTOR_POLICY_BLOB: cannot open %s\n", path);
                exit(1);
            }
            blob.resize(2084);
            size_t n = fread(blob.data(), sizeof(float), blob.size(), f);
            fclose(f);
            if (n != blob.size()) {
                fprintf(stderr, "RAPTOR_POLICY_BLOB: %s is not 2084 floats\n", path);
                exit(1);
            }
            printf("raptor: initialized actor from %s\n", path);
        }
    }
    return blob.empty() ? nullptr : blob.data();
}

static void rn_upload(PrecisionTensor* t, const float* src, cudaStream_t stream) {
    cudaMemcpyAsync(t->data, src, numel(t->shape) * sizeof(precision_t),
        cudaMemcpyHostToDevice, stream);
}

struct RaptorEncWeights {
    PrecisionTensor w, b;
    int in_dim, out_dim;
};

struct RaptorEncActs {
    PrecisionTensor out, saved_input, grad_pre, wgrad, bgrad;
};

__global__ void rn_bias_relu(precision_t* out, const precision_t* b, int n, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = to_float(out[i]) + to_float(b[i % H]);
    out[i] = from_float(v > 0 ? v : 0);
}

__global__ void rn_relu_mask(precision_t* dst, const precision_t* grad,
        const precision_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = from_float(to_float(out[i]) > 0 ? to_float(grad[i]) : 0.0f);
}

__global__ void rn_col_sum(precision_t* dst, const precision_t* src, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float s = 0;
    for (int r = 0; r < rows; r++) s += to_float(src[r * cols + c]);
    dst[c] = from_float(s);
}

__global__ void rn_add(precision_t* dst, const precision_t* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = from_float(to_float(dst[i]) + to_float(src[i]));
}

__global__ void rn_broadcast(precision_t* dst, const precision_t* h0, int H, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = h0[i % H];
}

__global__ void rn_reset_state(precision_t* h, const precision_t* done, const precision_t* h0,
        int H, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && to_float(done[i / H]) > 0.0f) h[i] = h0[i % H];
}

// a null dones resets every row; used to seed the buffers with h0 before epoch 0
__global__ void rn_reset_rows(precision_t* h, const float* done, const precision_t* h0,
        int H, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (done == nullptr || done[i / H] > 0.0f)) h[i] = h0[i % H];
}

// at a reset the state came from h0, so move that gradient out of the recurrence
__global__ void rn_split_reset_grad(precision_t* dh, precision_t* h0_accum,
        const precision_t* done, int H, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || to_float(done[i / H]) <= 0.0f) return;
    h0_accum[i] = from_float(to_float(h0_accum[i]) + to_float(dh[i]));
    dh[i] = from_float(0.0f);
}

static PrecisionTensor raptor_enc_forward(void* w, void* activations, PrecisionTensor input,
        cudaStream_t stream) {
    RaptorEncWeights* ew = (RaptorEncWeights*)w;
    RaptorEncActs* a = (RaptorEncActs*)activations;
    if (a->saved_input.data) puf_copy(&a->saved_input, &input, stream);
    puf_mm(&input, &ew->w, &a->out, stream);
    int n = numel(a->out.shape);
    rn_bias_relu<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->out.data, ew->b.data, n, ew->out_dim);
    return a->out;
}

static void raptor_enc_backward(void* w, void* activations, PrecisionTensor grad,
        cudaStream_t stream) {
    RaptorEncWeights* ew = (RaptorEncWeights*)w;
    RaptorEncActs* a = (RaptorEncActs*)activations;
    int n = numel(a->out.shape);
    rn_relu_mask<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->grad_pre.data, grad.data,
        a->out.data, n);
    puf_mm_tn(&a->grad_pre, &a->saved_input, &a->wgrad, stream);
    rn_col_sum<<<grid_size(ew->out_dim), BLOCK_SIZE, 0, stream>>>(a->bgrad.data,
        a->grad_pre.data, n / ew->out_dim, ew->out_dim);
}

static void raptor_enc_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    RaptorEncWeights* ew = (RaptorEncWeights*)w;
    const float* blob = rn_blob();
    if (blob) {
        rn_upload(&ew->w, blob + 0, stream);
        rn_upload(&ew->b, blob + 352, stream);
        return;
    }
    PrecisionTensor wt = {.data = ew->w.data, .shape = {ew->out_dim, ew->in_dim}};
    puf_kaiming_init(&wt, std::sqrt(2.0f), (*seed)++, stream);
    cudaMemsetAsync(ew->b.data, 0, numel(ew->b.shape) * sizeof(precision_t), stream);
}

static void raptor_enc_reg_params(void* w, Allocator* alloc) {
    RaptorEncWeights* ew = (RaptorEncWeights*)w;
    ew->w = {.shape = {ew->out_dim, ew->in_dim}};
    ew->b = {.shape = {ew->out_dim}};
    alloc_register(alloc, &ew->w);
    alloc_register(alloc, &ew->b);
}

static void raptor_enc_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads,
        int B_TT) {
    RaptorEncWeights* ew = (RaptorEncWeights*)w;
    RaptorEncActs* a = (RaptorEncActs*)activations;
    *a = (RaptorEncActs){
        .out = {.shape = {B_TT, ew->out_dim}},
        .saved_input = {.shape = {B_TT, ew->in_dim}},
        .grad_pre = {.shape = {B_TT, ew->out_dim}},
        .wgrad = {.shape = {ew->out_dim, ew->in_dim}},
        .bgrad = {.shape = {ew->out_dim}},
    };
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_input);
    alloc_register(acts, &a->grad_pre);
    alloc_register(grads, &a->wgrad);
    alloc_register(grads, &a->bgrad);
}

static void raptor_enc_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    RaptorEncWeights* ew = (RaptorEncWeights*)w;
    RaptorEncActs* a = (RaptorEncActs*)activations;
    *a = (RaptorEncActs){};
    a->out = {.shape = {B, ew->out_dim}};
    alloc_register(alloc, &a->out);
}

static void* raptor_enc_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    RaptorEncWeights* ew = (RaptorEncWeights*)calloc(1, sizeof(RaptorEncWeights));
    ew->in_dim = e->in_dim;
    ew->out_dim = e->out_dim;
    return ew;
}

static void raptor_free(void* p) { free(p); }

struct RaptorGRUWeights {
    PrecisionTensor wi, wh, bi, bh, h0;
    int hidden, horizon;
};

struct RaptorGRUActs {
    PrecisionTensor out, next_state, gi, gh;
    PrecisionTensor xt, gi_all, gh_all, hprev, r, z, ncand, ghn, outt;
    PrecisionTensor gradt, dgi, dgh, dgradt, grad_input, hcur, dh_carry;
    PrecisionTensor wi_grad, wh_grad, bi_grad, bh_grad, h0_grad;
    PrecisionTensor dones, donest, h0_accum;
    int B, TT;
    bool used_h0;
};

__global__ void rn_gru_gate(precision_t* out, precision_t* r_buf, precision_t* z_buf,
        precision_t* n_buf, precision_t* ghn_buf, const precision_t* gi, const precision_t* gh,
        const precision_t* state, const precision_t* bi, const precision_t* bh, int H, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    int b = idx / H, i = idx % H;
    const precision_t* gi_row = gi + b * RN_GATES * H;
    const precision_t* gh_row = gh + b * RN_GATES * H;
    float gh_n = to_float(gh_row[2 * H + i]) + to_float(bh[2 * H + i]);
    float r = sigmoid(to_float(gi_row[i]) + to_float(bi[i]) +
                      to_float(gh_row[i]) + to_float(bh[i]));
    float z = sigmoid(to_float(gi_row[H + i]) + to_float(bi[H + i]) +
                      to_float(gh_row[H + i]) + to_float(bh[H + i]));
    float n = tanhf(to_float(gi_row[2 * H + i]) + to_float(bi[2 * H + i]) + r * gh_n);
    float h_prev = to_float(state[idx]);
    out[idx] = from_float((1.0f - z) * n + z * h_prev);
    if (!r_buf) return;
    r_buf[idx] = from_float(r);
    z_buf[idx] = from_float(z);
    n_buf[idx] = from_float(n);
    ghn_buf[idx] = from_float(gh_n);
}

__global__ void rn_gru_gate_bwd(precision_t* dgi, precision_t* dgh, precision_t* dh_out,
        const precision_t* dh, const precision_t* r_buf, const precision_t* z_buf,
        const precision_t* n_buf, const precision_t* ghn_buf, const precision_t* hprev,
        int H, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    int b = idx / H, i = idx % H;
    float g = to_float(dh[idx]);
    float r = to_float(r_buf[idx]), z = to_float(z_buf[idx]), n = to_float(n_buf[idx]);
    float dn = g * (1.0f - z) * (1.0f - n * n);
    float dz = g * (to_float(hprev[idx]) - n) * z * (1.0f - z);
    float dr = dn * to_float(ghn_buf[idx]) * r * (1.0f - r);
    precision_t* dgi_row = dgi + b * RN_GATES * H;
    precision_t* dgh_row = dgh + b * RN_GATES * H;
    dgi_row[i] = from_float(dr);
    dgi_row[H + i] = from_float(dz);
    dgi_row[2 * H + i] = from_float(dn);
    dgh_row[i] = from_float(dr);
    dgh_row[H + i] = from_float(dz);
    dgh_row[2 * H + i] = from_float(dn * r);
    dh_out[idx] = from_float(g * z);
}

static PrecisionTensor rn_step(PrecisionTensor t, int t_idx, int B, int width) {
    PrecisionTensor s = {.data = t.data + (size_t)t_idx * B * width, .shape = {B, width}};
    return s;
}

static PrecisionTensor raptor_gru_forward(void* w, PrecisionTensor x, PrecisionTensor state,
        void* activations, cudaStream_t stream) {
    RaptorGRUWeights* m = (RaptorGRUWeights*)w;
    RaptorGRUActs* a = (RaptorGRUActs*)activations;
    int B = numel(x.shape) / m->hidden, H = m->hidden;
    puf_mm(&x, &m->wi, &a->gi, stream);
    puf_mm(&state, &m->wh, &a->gh, stream);
    rn_gru_gate<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->out.data, nullptr, nullptr,
        nullptr, nullptr, a->gi.data, a->gh.data, state.data, m->bi.data, m->bh.data, H, B);
    puf_copy(&state, &a->out, stream);
    return a->out;
}

static PrecisionTensor raptor_gru_forward_train(void* w, PrecisionTensor x, PrecisionTensor state,
        PrecisionTensor dones, void* activations, cudaStream_t stream) {
    RaptorGRUWeights* m = (RaptorGRUWeights*)w;
    RaptorGRUActs* a = (RaptorGRUActs*)activations;
    int B = a->B, TT = a->TT, H = m->hidden, G = RN_GATES * H;
    a->dones = dones;
    a->used_h0 = (state.data == nullptr);
    transpose_102<<<grid_size(B * TT * H), BLOCK_SIZE, 0, stream>>>(a->xt.data, x.data, B, TT, H);
    if (dones.data)
        transpose_102<<<grid_size(B * TT), BLOCK_SIZE, 0, stream>>>(a->donest.data, dones.data,
            B, TT, 1);
    PrecisionTensor xt_flat = {.data = a->xt.data, .shape = {TT * B, H}};
    puf_mm(&xt_flat, &m->wi, &a->gi_all, stream);

    if (state.data) {
        PrecisionTensor s = {.data = state.data, .shape = {B, H}};
        puf_copy(&a->hcur, &s, stream);
    } else {
        rn_broadcast<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->hcur.data, m->h0.data,
            H, B * H);
    }
    for (int t = 0; t < TT; t++) {
        if (dones.data)
            rn_reset_state<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->hcur.data,
                rn_step(a->donest, t, B, 1).data, m->h0.data, H, B * H);
        PrecisionTensor hprev_t = rn_step(a->hprev, t, B, H);
        puf_copy(&hprev_t, &a->hcur, stream);
        PrecisionTensor gh_t = rn_step(a->gh_all, t, B, G);
        puf_mm(&hprev_t, &m->wh, &gh_t, stream);
        PrecisionTensor out_t = rn_step(a->outt, t, B, H);
        rn_gru_gate<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(out_t.data,
            rn_step(a->r, t, B, H).data, rn_step(a->z, t, B, H).data,
            rn_step(a->ncand, t, B, H).data, rn_step(a->ghn, t, B, H).data,
            rn_step(a->gi_all, t, B, G).data, gh_t.data, hprev_t.data,
            m->bi.data, m->bh.data, H, B);
        puf_copy(&a->hcur, &out_t, stream);
    }
    transpose_102<<<grid_size(B * TT * H), BLOCK_SIZE, 0, stream>>>(a->out.data, a->outt.data,
        TT, B, H);
    return a->out;
}

static PrecisionTensor raptor_gru_backward(void* w, PrecisionTensor grad, void* activations,
        cudaStream_t stream) {
    RaptorGRUWeights* m = (RaptorGRUWeights*)w;
    RaptorGRUActs* a = (RaptorGRUActs*)activations;
    int B = a->B, TT = a->TT, H = m->hidden, G = RN_GATES * H;
    transpose_102<<<grid_size(B * TT * H), BLOCK_SIZE, 0, stream>>>(a->gradt.data, grad.data,
        B, TT, H);
    puf_zero(&a->hcur, stream);
    puf_zero(&a->h0_accum, stream);
    for (int t = TT - 1; t >= 0; t--) {
        rn_add<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->hcur.data,
            rn_step(a->gradt, t, B, H).data, B * H);
        rn_gru_gate_bwd<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
            rn_step(a->dgi, t, B, G).data, rn_step(a->dgh, t, B, G).data, a->dh_carry.data,
            a->hcur.data, rn_step(a->r, t, B, H).data, rn_step(a->z, t, B, H).data,
            rn_step(a->ncand, t, B, H).data, rn_step(a->ghn, t, B, H).data,
            rn_step(a->hprev, t, B, H).data, H, B);
        PrecisionTensor dgh_t = rn_step(a->dgh, t, B, G);
        puf_mm_nn(&dgh_t, &m->wh, &a->hcur, stream);
        rn_add<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->hcur.data, a->dh_carry.data,
            B * H);
        if (a->dones.data)
            rn_split_reset_grad<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->hcur.data,
                a->h0_accum.data, rn_step(a->donest, t, B, 1).data, H, B * H);
    }
    if (a->used_h0)
        rn_add<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(a->h0_accum.data, a->hcur.data,
            B * H);
    rn_col_sum<<<grid_size(H), BLOCK_SIZE, 0, stream>>>(a->h0_grad.data, a->h0_accum.data, B, H);
    PrecisionTensor dgi_flat = {.data = a->dgi.data, .shape = {TT * B, G}};
    PrecisionTensor dgh_flat = {.data = a->dgh.data, .shape = {TT * B, G}};
    PrecisionTensor xt_flat = {.data = a->xt.data, .shape = {TT * B, H}};
    PrecisionTensor hprev_flat = {.data = a->hprev.data, .shape = {TT * B, H}};
    puf_mm_tn(&dgi_flat, &xt_flat, &a->wi_grad, stream);
    puf_mm_tn(&dgh_flat, &hprev_flat, &a->wh_grad, stream);
    rn_col_sum<<<grid_size(G), BLOCK_SIZE, 0, stream>>>(a->bi_grad.data, a->dgi.data, TT * B, G);
    rn_col_sum<<<grid_size(G), BLOCK_SIZE, 0, stream>>>(a->bh_grad.data, a->dgh.data, TT * B, G);
    puf_mm_nn(&dgi_flat, &m->wi, &a->dgradt, stream);
    transpose_102<<<grid_size(B * TT * H), BLOCK_SIZE, 0, stream>>>(a->grad_input.data,
        a->dgradt.data, TT, B, H);
    return a->grad_input;
}

static void raptor_gru_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    RaptorGRUWeights* m = (RaptorGRUWeights*)w;
    const float* blob = rn_blob();
    if (blob) {
        rn_upload(&m->wi, blob + 368, stream);
        rn_upload(&m->wh, blob + 1136, stream);
        rn_upload(&m->bi, blob + 1904, stream);
        rn_upload(&m->bh, blob + 1952, stream);
        rn_upload(&m->h0, blob + 2000, stream);
        return;
    }
    PrecisionTensor wi = {.data = m->wi.data, .shape = {RN_GATES * m->hidden, m->hidden}};
    PrecisionTensor wh = {.data = m->wh.data, .shape = {RN_GATES * m->hidden, m->hidden}};
    puf_kaiming_init(&wi, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&wh, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(m->bi.data, 0, numel(m->bi.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(m->bh.data, 0, numel(m->bh.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(m->h0.data, 0, numel(m->h0.shape) * sizeof(precision_t), stream);
}

static void raptor_gru_reg_params(void* w, Allocator* alloc) {
    RaptorGRUWeights* m = (RaptorGRUWeights*)w;
    int H = m->hidden, G = RN_GATES * H;
    m->wi = {.shape = {G, H}};
    m->wh = {.shape = {G, H}};
    m->bi = {.shape = {G}};
    m->bh = {.shape = {G}};
    m->h0 = {.shape = {H}};
    for (PrecisionTensor* t : {&m->wi, &m->wh, &m->bi, &m->bh, &m->h0}) alloc_register(alloc, t);
}

static void raptor_gru_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads,
        int B_TT) {
    RaptorGRUWeights* m = (RaptorGRUWeights*)w;
    RaptorGRUActs* a = (RaptorGRUActs*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT, G = RN_GATES * H;
    *a = (RaptorGRUActs){};
    a->B = B;
    a->TT = TT;
    a->out = {.shape = {B, TT, H}};
    a->grad_input = {.shape = {B, TT, H}};
    for (PrecisionTensor* t : {&a->xt, &a->hprev, &a->r, &a->z, &a->ncand, &a->ghn, &a->outt,
            &a->gradt, &a->dgradt})
        *t = {.shape = {TT, B, H}};
    for (PrecisionTensor* t : {&a->gi_all, &a->gh_all, &a->dgi, &a->dgh})
        *t = {.shape = {TT, B, G}};
    a->hcur = {.shape = {B, H}};
    a->dh_carry = {.shape = {B, H}};
    a->h0_accum = {.shape = {B, H}};
    a->donest = {.shape = {TT, B, 1}};
    for (PrecisionTensor* t : {&a->out, &a->grad_input, &a->xt, &a->hprev, &a->r, &a->z,
            &a->ncand, &a->ghn, &a->outt, &a->gradt, &a->dgradt, &a->gi_all, &a->gh_all,
            &a->dgi, &a->dgh, &a->hcur, &a->dh_carry, &a->h0_accum, &a->donest})
        alloc_register(acts, t);
    a->wi_grad = {.shape = {G, H}};
    a->wh_grad = {.shape = {G, H}};
    a->bi_grad = {.shape = {G}};
    a->bh_grad = {.shape = {G}};
    a->h0_grad = {.shape = {H}};
    for (PrecisionTensor* t : {&a->wi_grad, &a->wh_grad, &a->bi_grad, &a->bh_grad, &a->h0_grad})
        alloc_register(grads, t);
}

static void raptor_gru_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    RaptorGRUWeights* m = (RaptorGRUWeights*)w;
    RaptorGRUActs* a = (RaptorGRUActs*)activations;
    int H = m->hidden;
    *a = (RaptorGRUActs){};
    a->out = {.shape = {B, H}};
    a->gi = {.shape = {B, RN_GATES * H}};
    a->gh = {.shape = {B, RN_GATES * H}};
    for (PrecisionTensor* t : {&a->out, &a->gi, &a->gh}) alloc_register(alloc, t);
}

static void* raptor_gru_create_weights(void* self) {
    Network* n = (Network*)self;
    RaptorGRUWeights* m = (RaptorGRUWeights*)calloc(1, sizeof(RaptorGRUWeights));
    m->hidden = n->hidden;
    m->horizon = n->horizon;
    return m;
}

static void create_raptor_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = raptor_enc_forward,
        .backward = raptor_enc_backward,
        .init_weights = raptor_enc_init_weights,
        .reg_params = raptor_enc_reg_params,
        .reg_train = raptor_enc_reg_train,
        .reg_rollout = raptor_enc_reg_rollout,
        .create_weights = raptor_enc_create_weights,
        .free_weights = raptor_free,
        .free_activations = raptor_free,
        .in_dim = enc->in_dim,
        .out_dim = enc->out_dim,
        .activation_size = sizeof(RaptorEncActs),
    };
}

static void raptor_gru_reset(void* w, PrecisionTensor state, const float* dones, int row_off,
        int rows, cudaStream_t stream) {
    RaptorGRUWeights* m = (RaptorGRUWeights*)w;
    int H = m->hidden, n = rows * H;
    rn_reset_rows<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)state.data + (long)row_off * H, dones, m->h0.data, H, n);
}

static void create_raptor_network(Network* net) {
    *net = Network{
        .reset = raptor_gru_reset,
        .forward = raptor_gru_forward,
        .forward_train = raptor_gru_forward_train,
        .backward = raptor_gru_backward,
        .init_weights = raptor_gru_init_weights,
        .reg_params = raptor_gru_reg_params,
        .reg_train = raptor_gru_reg_train,
        .reg_rollout = raptor_gru_reg_rollout,
        .create_weights = raptor_gru_create_weights,
        .free_weights = raptor_free,
        .free_activations = raptor_free,
        .hidden = net->hidden,
        .num_layers = net->num_layers,
        .horizon = net->horizon,
        .activation_size = sizeof(RaptorGRUActs),
    };
}

static constexpr int RN_OBS = 22, RN_CRITIC = 256, RN_HID = 16, RN_CIN = RN_OBS + RN_HID;

__global__ void rn_concat(precision_t* dst, const precision_t* h, const precision_t* obs,
        int H, int O, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = i / (H + O), j = i % (H + O);
    dst[i] = (j < H) ? h[b * H + j] : obs[b * O + j - H];
}

// prefix must mirror DecoderWeights: pufferlib.cu casts every decoder to it for logstd
struct RaptorDecWeights {
    PrecisionTensor w, logstd;
    int hidden_dim, output_dim;
    bool continuous;
    PrecisionTensor b, c1w, c1b, c2w, c2b, c3w, c3b;
};

struct RaptorDecActs {
    PrecisionTensor out, act, val, k1, k2, saved_h, cin;
    PrecisionTensor dact, dval, dk1, dk2, grad_input;
    PrecisionTensor wgrad, bgrad, logstd_grad, c1wgrad, c1bgrad, c2wgrad, c2bgrad, c3wgrad, c3bgrad;
};

__global__ void rn_bias(precision_t* out, const precision_t* b, int n, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = from_float(to_float(out[i]) + to_float(b[i % H]));
}

__global__ void rn_bias_tanh(precision_t* out, const precision_t* b, int n, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = from_float(tanhf(to_float(out[i]) + to_float(b[i % H])));
}

__global__ void rn_tanh_bwd(precision_t* dst, const precision_t* grad, const precision_t* out,
        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float o = to_float(out[i]);
    dst[i] = from_float(to_float(grad[i]) * (1.0f - o * o));
}

__global__ void rn_assemble(precision_t* out, const precision_t* act, const precision_t* val,
        int B, int A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * (A + 1)) return;
    int r = i / (A + 1), c = i % (A + 1);
    out[i] = (c < A) ? act[r * A + c] : val[r];
}

__global__ void rn_split_grad(precision_t* dact, precision_t* dval, const float* grad_logits,
        const float* grad_value, int B, int A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * A) dact[i] = from_float(grad_logits[i]);
    if (i < B) dval[i] = from_float(grad_value[i]);
}

static PrecisionTensor raptor_dec_forward(void* w, void* activations, PrecisionTensor input,
        PrecisionTensor obs, cudaStream_t stream) {
    RaptorDecWeights* dw = (RaptorDecWeights*)w;
    RaptorDecActs* a = (RaptorDecActs*)activations;
    int B = numel(input.shape) / dw->hidden_dim, A = dw->output_dim;
    int H = dw->hidden_dim;
    if (a->saved_h.data) puf_copy(&a->saved_h, &input, stream);
    puf_mm(&input, &dw->w, &a->act, stream);
    rn_bias<<<grid_size(B * A), BLOCK_SIZE, 0, stream>>>(a->act.data, dw->b.data, B * A, A);
    // detached: the concat is a fresh buffer, so no critic gradient reaches the GRU
    rn_concat<<<grid_size(B * (H + RN_OBS)), BLOCK_SIZE, 0, stream>>>(a->cin.data, input.data,
        obs.data, H, RN_OBS, B * (H + RN_OBS));
    puf_mm(&a->cin, &dw->c1w, &a->k1, stream);
    rn_bias_tanh<<<grid_size(B * RN_CRITIC), BLOCK_SIZE, 0, stream>>>(a->k1.data, dw->c1b.data,
        B * RN_CRITIC, RN_CRITIC);
    puf_mm(&a->k1, &dw->c2w, &a->k2, stream);
    rn_bias_tanh<<<grid_size(B * RN_CRITIC), BLOCK_SIZE, 0, stream>>>(a->k2.data, dw->c2b.data,
        B * RN_CRITIC, RN_CRITIC);
    puf_mm(&a->k2, &dw->c3w, &a->val, stream);
    rn_bias<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(a->val.data, dw->c3b.data, B, 1);
    rn_assemble<<<grid_size(B * (A + 1)), BLOCK_SIZE, 0, stream>>>(a->out.data, a->act.data,
        a->val.data, B, A);
    return a->out;
}

static PrecisionTensor raptor_dec_backward(void* w, void* activations, FloatTensor grad_logits,
        FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream) {
    RaptorDecWeights* dw = (RaptorDecWeights*)w;
    RaptorDecActs* a = (RaptorDecActs*)activations;
    int A = dw->output_dim, B = grad_logits.shape[0];
    rn_split_grad<<<grid_size(B * A), BLOCK_SIZE, 0, stream>>>(a->dact.data, a->dval.data,
        grad_logits.data, grad_value.data, B, A);
    puf_mm_tn(&a->dact, &a->saved_h, &a->wgrad, stream);
    rn_col_sum<<<grid_size(A), BLOCK_SIZE, 0, stream>>>(a->bgrad.data, a->dact.data, B, A);
    puf_mm_nn(&a->dact, &dw->w, &a->grad_input, stream);
    if (dw->continuous && grad_logstd.data)
        sum_rows_to_precision_kernel<<<grid_size(A), BLOCK_SIZE, 0, stream>>>(
            a->logstd_grad.data, grad_logstd.data, B, A);

    puf_mm_tn(&a->dval, &a->k2, &a->c3wgrad, stream);
    rn_col_sum<<<grid_size(1), BLOCK_SIZE, 0, stream>>>(a->c3bgrad.data, a->dval.data, B, 1);
    puf_mm_nn(&a->dval, &dw->c3w, &a->dk2, stream);
    rn_tanh_bwd<<<grid_size(B * RN_CRITIC), BLOCK_SIZE, 0, stream>>>(a->dk2.data, a->dk2.data,
        a->k2.data, B * RN_CRITIC);
    puf_mm_tn(&a->dk2, &a->k1, &a->c2wgrad, stream);
    rn_col_sum<<<grid_size(RN_CRITIC), BLOCK_SIZE, 0, stream>>>(a->c2bgrad.data, a->dk2.data, B,
        RN_CRITIC);
    puf_mm_nn(&a->dk2, &dw->c2w, &a->dk1, stream);
    rn_tanh_bwd<<<grid_size(B * RN_CRITIC), BLOCK_SIZE, 0, stream>>>(a->dk1.data, a->dk1.data,
        a->k1.data, B * RN_CRITIC);
    puf_mm_tn(&a->dk1, &a->cin, &a->c1wgrad, stream);
    rn_col_sum<<<grid_size(RN_CRITIC), BLOCK_SIZE, 0, stream>>>(a->c1bgrad.data, a->dk1.data, B,
        RN_CRITIC);
    return a->grad_input;
}

static void raptor_dec_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    RaptorDecWeights* dw = (RaptorDecWeights*)w;
    const float* blob = rn_blob();
    puf_kaiming_init(&dw->c1w, std::sqrt(2.0f), (*seed)++, stream);
    puf_kaiming_init(&dw->c2w, std::sqrt(2.0f), (*seed)++, stream);
    puf_kaiming_init(&dw->c3w, 1.0f, (*seed)++, stream);
    for (PrecisionTensor* t : {&dw->c1b, &dw->c2b, &dw->c3b})
        cudaMemsetAsync(t->data, 0, numel(t->shape) * sizeof(precision_t), stream);
    fill_precision_kernel<<<grid_size(dw->output_dim), BLOCK_SIZE, 0, stream>>>(
        dw->logstd.data, from_float(RN_LOGSTD_INIT), dw->output_dim);
    if (blob) {
        rn_upload(&dw->w, blob + 2016, stream);
        rn_upload(&dw->b, blob + 2080, stream);
        return;
    }
    PrecisionTensor aw = {.data = dw->w.data, .shape = {dw->output_dim, dw->hidden_dim}};
    puf_kaiming_init(&aw, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(dw->b.data, 0, numel(dw->b.shape) * sizeof(precision_t), stream);
}

static void raptor_dec_reg_params(void* w, Allocator* alloc) {
    RaptorDecWeights* dw = (RaptorDecWeights*)w;
    dw->w = {.shape = {dw->output_dim, dw->hidden_dim}};
    dw->b = {.shape = {dw->output_dim}};
    alloc_register(alloc, &dw->w);
    alloc_register(alloc, &dw->b);
    if (dw->continuous) {
        dw->logstd = {.shape = {1, dw->output_dim}};
        alloc_register(alloc, &dw->logstd);
    }
    dw->c1w = {.shape = {RN_CRITIC, RN_CIN}};
    dw->c1b = {.shape = {RN_CRITIC}};
    dw->c2w = {.shape = {RN_CRITIC, RN_CRITIC}};
    dw->c2b = {.shape = {RN_CRITIC}};
    dw->c3w = {.shape = {1, RN_CRITIC}};
    dw->c3b = {.shape = {1}};
    for (PrecisionTensor* t : {&dw->c1w, &dw->c1b, &dw->c2w, &dw->c2b, &dw->c3w, &dw->c3b})
        alloc_register(alloc, t);
}

static void raptor_dec_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads,
        int B_TT) {
    RaptorDecWeights* dw = (RaptorDecWeights*)w;
    RaptorDecActs* a = (RaptorDecActs*)activations;
    int A = dw->output_dim, H = dw->hidden_dim;
    *a = (RaptorDecActs){};
    a->out = {.shape = {B_TT, A + 1}};
    a->act = {.shape = {B_TT, A}};
    a->val = {.shape = {B_TT, 1}};
    a->k1 = {.shape = {B_TT, RN_CRITIC}};
    a->k2 = {.shape = {B_TT, RN_CRITIC}};
    a->saved_h = {.shape = {B_TT, H}};
    a->cin = {.shape = {B_TT, RN_CIN}};
    a->dact = {.shape = {B_TT, A}};
    a->dval = {.shape = {B_TT, 1}};
    a->dk1 = {.shape = {B_TT, RN_CRITIC}};
    a->dk2 = {.shape = {B_TT, RN_CRITIC}};
    a->grad_input = {.shape = {B_TT, H}};
    for (PrecisionTensor* t : {&a->out, &a->act, &a->val, &a->k1, &a->k2, &a->saved_h,
            &a->cin, &a->dact, &a->dval, &a->dk1, &a->dk2, &a->grad_input})
        alloc_register(acts, t);
    a->wgrad = {.shape = {A, H}};
    a->bgrad = {.shape = {A}};
    a->logstd_grad = {.shape = {1, A}};
    a->c1wgrad = {.shape = {RN_CRITIC, RN_CIN}};
    a->c1bgrad = {.shape = {RN_CRITIC}};
    a->c2wgrad = {.shape = {RN_CRITIC, RN_CRITIC}};
    a->c2bgrad = {.shape = {RN_CRITIC}};
    a->c3wgrad = {.shape = {1, RN_CRITIC}};
    a->c3bgrad = {.shape = {1}};
    alloc_register(grads, &a->wgrad);
    alloc_register(grads, &a->bgrad);
    if (dw->continuous) alloc_register(grads, &a->logstd_grad);
    for (PrecisionTensor* t : {&a->c1wgrad, &a->c1bgrad, &a->c2wgrad, &a->c2bgrad, &a->c3wgrad,
            &a->c3bgrad})
        alloc_register(grads, t);
}

static void raptor_dec_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    RaptorDecWeights* dw = (RaptorDecWeights*)w;
    RaptorDecActs* a = (RaptorDecActs*)activations;
    *a = (RaptorDecActs){};
    a->out = {.shape = {B, dw->output_dim + 1}};
    a->act = {.shape = {B, dw->output_dim}};
    a->val = {.shape = {B, 1}};
    a->k1 = {.shape = {B, RN_CRITIC}};
    a->k2 = {.shape = {B, RN_CRITIC}};
    a->cin = {.shape = {B, RN_CIN}};
    for (PrecisionTensor* t : {&a->out, &a->act, &a->val, &a->k1, &a->k2, &a->cin})
        alloc_register(alloc, t);
}

static void* raptor_dec_create_weights(void* self) {
    Decoder* d = (Decoder*)self;
    RaptorDecWeights* dw = (RaptorDecWeights*)calloc(1, sizeof(RaptorDecWeights));
    dw->hidden_dim = d->hidden_dim;
    dw->output_dim = d->output_dim;
    dw->continuous = d->continuous;
    return dw;
}

static void create_raptor_decoder(Decoder* dec) {
    *dec = Decoder{
        .forward = raptor_dec_forward,
        .backward = raptor_dec_backward,
        .init_weights = raptor_dec_init_weights,
        .reg_params = raptor_dec_reg_params,
        .reg_train = raptor_dec_reg_train,
        .reg_rollout = raptor_dec_reg_rollout,
        .create_weights = raptor_dec_create_weights,
        .free_weights = raptor_free,
        .free_activations = raptor_free,
        .hidden_dim = dec->hidden_dim,
        .output_dim = dec->output_dim,
        .continuous = dec->continuous,
        .activation_size = (int)sizeof(RaptorDecActs),
    };
}
