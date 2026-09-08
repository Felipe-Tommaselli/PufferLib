// Forward parity of the CUDA modules against the shipped example fixtures.
// nvcc -O2 -std=c++17 -I../../src -DPRECISION_FLOAT net_test.cu -o net_test -lcublas

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "models.cu"
#include "raptor_net.cu"
#include "muon.cu"

static constexpr int OBS = 22, HID = 16, ACT = 4;

static std::vector<float> read_file(const char* path, size_t offset_bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f) - offset_bytes;
    fseek(f, offset_bytes, SEEK_SET);
    std::vector<float> v(bytes / sizeof(float));
    size_t n = fread(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    if (n != v.size()) exit(1);
    return v;
}

static void upload(PrecisionTensor* t, const float* src) {
    cudaMemcpy(t->data, src, numel(t->shape) * sizeof(float), cudaMemcpyHostToDevice);
}

static int parity() {
    std::vector<float> pw = read_file("../../../../artifacts/raptor/base_policy.bin", 0);
    std::vector<float> fin = read_file("../../../../artifacts/raptor/fixture_in.bin", 8);
    std::vector<float> fout = read_file("../../../../artifacts/raptor/fixture_out.bin", 8);
    int TT = 500, B = 2;

    Encoder enc = {.in_dim = OBS, .out_dim = HID};
    create_raptor_encoder(&enc);
    Network net = {.hidden = HID, .num_layers = 1, .horizon = TT};
    create_raptor_network(&net);
    void* ew = enc.create_weights(&enc);
    void* nw = net.create_weights(&net);

    Allocator params = {}, acts = {}, grads = {};
    enc.reg_params(ew, &params);
    net.reg_params(nw, &params);
    void* enc_a = calloc(1, enc.activation_size);
    void* net_a = calloc(1, net.activation_size);
    enc.reg_train(ew, enc_a, &acts, &grads, B * TT);
    net.reg_train(nw, net_a, &acts, &grads, B * TT);
    alloc_create(&params);
    alloc_create(&acts);
    alloc_create(&grads);

    RaptorEncWeights* e = (RaptorEncWeights*)ew;
    RaptorGRUWeights* g = (RaptorGRUWeights*)nw;
    upload(&e->w, pw.data() + 0);
    upload(&e->b, pw.data() + 352);
    upload(&g->wi, pw.data() + 368);
    upload(&g->wh, pw.data() + 1136);
    upload(&g->bi, pw.data() + 1904);
    upload(&g->bh, pw.data() + 1952);
    upload(&g->h0, pw.data() + 2000);

    // fixtures are time-major; the trainer hands the policy (B, TT, feat)
    std::vector<float> obs(B * TT * OBS);
    for (int t = 0; t < TT; t++)
        for (int b = 0; b < B; b++)
            for (int k = 0; k < OBS; k++)
                obs[(b * TT + t) * OBS + k] = fin[(t * B + b) * OBS + k];

    PrecisionTensor x = {.shape = {B * TT, OBS}};
    cudaMalloc(&x.data, obs.size() * sizeof(float));
    cudaMemcpy(x.data, obs.data(), obs.size() * sizeof(float), cudaMemcpyHostToDevice);

    PrecisionTensor h = enc.forward(ew, enc_a, x, 0);
    PrecisionTensor h3 = {.data = h.data, .shape = {B, TT, HID}};
    PrecisionTensor state = {}, dones = {};
    PrecisionTensor out = net.forward_train(nw, h3, state, dones, net_a, 0);
    cudaDeviceSynchronize();

    std::vector<float> hid(B * TT * HID);
    cudaMemcpy(hid.data(), out.data, hid.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float worst = 0;
    for (int t = 0; t < TT; t++) {
        for (int b = 0; b < B; b++) {
            for (int k = 0; k < ACT; k++) {
                float v = pw[2080 + k];
                for (int j = 0; j < HID; j++) {
                    v += pw[2016 + k * HID + j] * hid[(b * TT + t) * HID + j];
                }
                float e = fabsf(v - fout[(t * B + b) * ACT + k]);
                if (e > worst) worst = e;
            }
        }
    }
    printf("max|CUDA - shipped fixture| = %.3e over %d steps x %d agents\n", worst, TT, B);
    printf("%s\n", worst < 1e-4f ? "PASS" : "FAIL");
    return worst >= 1e-4f;
}

// The trainer acts through net.forward one step at a time; forward_train only runs in BPTT.
static int parity_rollout() {
    std::vector<float> pw = read_file("../../../../artifacts/raptor/base_policy.bin", 0);
    std::vector<float> fin = read_file("../../../../artifacts/raptor/fixture_in.bin", 8);
    std::vector<float> fout = read_file("../../../../artifacts/raptor/fixture_out.bin", 8);
    int TT = 500, B = 2;

    Encoder enc = {.in_dim = OBS, .out_dim = HID};
    create_raptor_encoder(&enc);
    Network net = {.hidden = HID, .num_layers = 1, .horizon = TT};
    create_raptor_network(&net);
    void* ew = enc.create_weights(&enc);
    void* nw = net.create_weights(&net);

    Allocator params = {}, acts = {};
    enc.reg_params(ew, &params);
    net.reg_params(nw, &params);
    void* enc_a = calloc(1, enc.activation_size);
    void* net_a = calloc(1, net.activation_size);
    enc.reg_rollout(ew, enc_a, &acts, B);
    net.reg_rollout(nw, net_a, &acts, B);
    PrecisionTensor state = {.shape = {B, HID}};
    PrecisionTensor bootstrap_state = {.shape = {B, HID}};
    alloc_register(&acts, &bootstrap_state);
    alloc_register(&acts, &state);
    alloc_create(&params);
    alloc_create(&acts);

    RaptorEncWeights* e = (RaptorEncWeights*)ew;
    RaptorGRUWeights* g = (RaptorGRUWeights*)nw;
    upload(&e->w, pw.data() + 0);
    upload(&e->b, pw.data() + 352);
    upload(&g->wi, pw.data() + 368);
    upload(&g->wh, pw.data() + 1136);
    upload(&g->bi, pw.data() + 1904);
    upload(&g->bh, pw.data() + 1952);
    upload(&g->h0, pw.data() + 2000);

    net.reset(nw, state, nullptr, 0, B, 0);

    PrecisionTensor x = {.shape = {B, OBS}};
    cudaMalloc(&x.data, (size_t)B * OBS * sizeof(float));
    std::vector<float> hid(B * HID), bootstrap(B * HID);
    float worst = 0;
    for (int t = 0; t < TT; t++) {
        if (t == TT / 2) net.reset(nw, state, nullptr, 0, B, 0);
        int fixture_t = t % (TT / 2);
        cudaMemcpy(x.data, fin.data() + (size_t)fixture_t * B * OBS, B * OBS * sizeof(float),
                   cudaMemcpyHostToDevice);
        PrecisionTensor h = enc.forward(ew, enc_a, x, 0);
        puf_copy(&bootstrap_state, &state, 0);
        PrecisionTensor probe = net.forward(nw, h, bootstrap_state, net_a, 0);
        cudaMemcpy(bootstrap.data(), probe.data, bootstrap.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        PrecisionTensor out = net.forward(nw, h, state, net_a, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(hid.data(), out.data, hid.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < B * HID; i++)
            worst = fmaxf(worst, fabsf(bootstrap[i] - hid[i]));
        for (int b = 0; b < B; b++) {
            for (int k = 0; k < ACT; k++) {
                float v = pw[2080 + k];
                for (int j = 0; j < HID; j++) v += pw[2016 + k * HID + j] * hid[b * HID + j];
                float err = fabsf(v - fout[(fixture_t * B + b) * ACT + k]);
                if (err > worst) worst = err;
            }
        }
    }
    printf("max|CUDA rollout/reset/copied bootstrap - fixture| = %.3e over %d steps x %d agents\n", worst, TT, B);
    printf("%s\n", worst < 1e-4f ? "PASS" : "FAIL");
    return worst >= 1e-4f;
}

// L = sum(mask * gru_out); analytic gradients must match central differences.
static int gradcheck(bool with_dones) {
    int TT = 8, B = 4;
    std::vector<float> pw = read_file("../../../../artifacts/raptor/base_policy.bin", 0);
    Encoder enc = {.in_dim = OBS, .out_dim = HID};
    create_raptor_encoder(&enc);
    Network net = {.hidden = HID, .num_layers = 1, .horizon = TT};
    create_raptor_network(&net);
    void* ew = enc.create_weights(&enc);
    void* nw = net.create_weights(&net);
    Allocator params = {}, acts = {}, grads = {};
    enc.reg_params(ew, &params);
    net.reg_params(nw, &params);
    void* enc_a = calloc(1, enc.activation_size);
    void* net_a = calloc(1, net.activation_size);
    enc.reg_train(ew, enc_a, &acts, &grads, B * TT);
    net.reg_train(nw, net_a, &acts, &grads, B * TT);
    alloc_create(&params);
    alloc_create(&acts);
    alloc_create(&grads);

    RaptorEncWeights* e = (RaptorEncWeights*)ew;
    RaptorGRUWeights* g = (RaptorGRUWeights*)nw;
    upload(&e->w, pw.data() + 0);
    upload(&e->b, pw.data() + 352);
    upload(&g->wi, pw.data() + 368);
    upload(&g->wh, pw.data() + 1136);
    upload(&g->bi, pw.data() + 1904);
    upload(&g->bh, pw.data() + 1952);
    upload(&g->h0, pw.data() + 2000);

    srand(7);
    std::vector<float> obs(B * TT * OBS), mask(B * TT * HID), host(B * TT * HID);
    for (float& v : obs) v = 2.0f * rand() / RAND_MAX - 1.0f;
    for (float& v : mask) v = 2.0f * rand() / RAND_MAX - 1.0f;
    PrecisionTensor x = {.shape = {B * TT, OBS}};
    cudaMalloc(&x.data, obs.size() * sizeof(float));
    cudaMemcpy(x.data, obs.data(), obs.size() * sizeof(float), cudaMemcpyHostToDevice);
    PrecisionTensor gmask = {.shape = {B, TT, HID}};
    cudaMalloc(&gmask.data, mask.size() * sizeof(float));
    cudaMemcpy(gmask.data, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice);

    // a reset mid-segment: the recurrence must restart from h0 and route gradient there
    std::vector<float> dn_host(B * TT, 0.0f);
    dn_host[0 * TT + 3] = 1.0f;
    dn_host[2 * TT + 5] = 1.0f;
    PrecisionTensor dones = {.shape = {B, TT}};
    cudaMalloc(&dones.data, dn_host.size() * sizeof(float));
    cudaMemcpy(dones.data, dn_host.data(), dn_host.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    if (!with_dones) dones = PrecisionTensor{};

    auto loss = [&]() {
        PrecisionTensor h = enc.forward(ew, enc_a, x, 0);
        PrecisionTensor h3 = {.data = h.data, .shape = {B, TT, HID}};
        PrecisionTensor state = {};
        PrecisionTensor out = net.forward_train(nw, h3, state, dones, net_a, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(host.data(), out.data, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double s = 0;
        for (size_t i = 0; i < host.size(); i++) s += (double)host[i] * mask[i];
        return s;
    };

    loss();
    PrecisionTensor dh = net.backward(nw, gmask, net_a, 0);
    enc.backward(ew, enc_a, dh, 0);
    cudaDeviceSynchronize();

    RaptorEncActs* ea = (RaptorEncActs*)enc_a;
    RaptorGRUActs* ga = (RaptorGRUActs*)net_a;
    struct Item { const char* name; PrecisionTensor* w; PrecisionTensor* grad; };
    Item items[] = {
        {"enc.w", &e->w, &ea->wgrad}, {"enc.b", &e->b, &ea->bgrad},
        {"gru.wi", &g->wi, &ga->wi_grad}, {"gru.wh", &g->wh, &ga->wh_grad},
        {"gru.bi", &g->bi, &ga->bi_grad}, {"gru.bh", &g->bh, &ga->bh_grad},
        {"gru.h0", &g->h0, &ga->h0_grad},
    };

    int fails = 0;
    float eps = 1e-3f;
    for (Item& it : items) {
        int total = numel(it.w->shape);
        std::vector<float> gr(total);
        cudaMemcpy(gr.data(), it.grad->data, total * sizeof(float), cudaMemcpyDeviceToHost);
        float worst = 0;
        for (int s = 0; s < 8; s++) {
            int i = (s * 7919 + 13) % total;
            float orig, up, dn;
            cudaMemcpy(&orig, it.w->data + i, sizeof(float), cudaMemcpyDeviceToHost);
            up = orig + eps;
            dn = orig - eps;
            cudaMemcpy(it.w->data + i, &up, sizeof(float), cudaMemcpyHostToDevice);
            double lp = loss();
            cudaMemcpy(it.w->data + i, &dn, sizeof(float), cudaMemcpyHostToDevice);
            double lm = loss();
            cudaMemcpy(it.w->data + i, &orig, sizeof(float), cudaMemcpyHostToDevice);
            float fd = (lp - lm) / (2 * eps);
            float rel = fabsf(fd - gr[i]) / fmaxf(1.0f, fmaxf(fabsf(fd), fabsf(gr[i])));
            if (rel > worst) worst = rel;
        }
        fails += worst > 2e-2f;
        printf("gradcheck %-7s %-5s max relative error = %.3e  %s\n", it.name,
               with_dones ? "reset" : "plain", worst, worst <= 2e-2f ? "PASS" : "FAIL");
    }
    return fails;
}

// Muon updates the flat weight buffer with the flat gradient buffer positionally, so the
// k-th gradient must sit at the same offset and size as the k-th parameter.
static int layout(bool emit = false) {
    int TT = 8, B = 4;
    Encoder enc = {.in_dim = OBS, .out_dim = HID};
    create_raptor_encoder(&enc);
    Decoder dec = {.hidden_dim = HID, .output_dim = ACT, .continuous = true};
    create_raptor_decoder(&dec);
    Network net = {.hidden = HID, .num_layers = 1, .horizon = TT};
    create_raptor_network(&net);
    void* ew = enc.create_weights(&enc);
    void* dwp = dec.create_weights(&dec);
    void* nw = net.create_weights(&net);

    Allocator params = {}, acts = {}, grads = {};
    enc.reg_params(ew, &params);
    dec.reg_params(dwp, &params);
    net.reg_params(nw, &params);
    void* enc_a = calloc(1, enc.activation_size);
    void* dec_a = calloc(1, dec.activation_size);
    void* net_a = calloc(1, net.activation_size);
    enc.reg_train(ew, enc_a, &acts, &grads, B * TT);
    dec.reg_train(dwp, dec_a, &acts, &grads, B * TT);
    net.reg_train(nw, net_a, &acts, &grads, B * TT);
    alloc_create(&params);
    alloc_create(&acts);
    alloc_create(&grads);

    RaptorEncWeights* e = (RaptorEncWeights*)ew;
    RaptorDecWeights* d = (RaptorDecWeights*)dwp;
    RaptorGRUWeights* g = (RaptorGRUWeights*)nw;
    RaptorEncActs* ea = (RaptorEncActs*)enc_a;
    RaptorDecActs* da = (RaptorDecActs*)dec_a;
    RaptorGRUActs* ga = (RaptorGRUActs*)net_a;
    PrecisionTensor* pt[] = {&e->w, &e->b, &d->w, &d->b, &d->logstd, &d->c1w, &d->c1b, &d->c2w,
        &d->c2b, &d->c3w, &d->c3b, &g->wi, &g->wh, &g->bi, &g->bh, &g->h0};
    PrecisionTensor* gt[] = {&ea->wgrad, &ea->bgrad, &da->wgrad, &da->bgrad, &da->logstd_grad,
        &da->c1wgrad, &da->c1bgrad, &da->c2wgrad, &da->c2bgrad, &da->c3wgrad, &da->c3bgrad,
        &ga->wi_grad, &ga->wh_grad, &ga->bi_grad, &ga->bh_grad, &ga->h0_grad};
    int n = sizeof(pt) / sizeof(pt[0]), bad = 0;
    for (int i = 0; i < n; i++) {
        long po = pt[i]->data - (precision_t*)params.mem;
        long go = gt[i]->data - (precision_t*)grads.mem;
        if (po != go || numel(pt[i]->shape) != numel(gt[i]->shape)) bad++;
    }
    printf("layout    %d/%d parameter and gradient slots aligned  %s\n", n - bad, n,
           bad ? "FAIL" : "PASS");
    printf("          %ld trainable parameters; actor slices enc.w=%ld enc.b=%ld dec.w=%ld "
           "dec.b=%ld gru.wi=%ld gru.wh=%ld gru.bi=%ld gru.bh=%ld gru.h0=%ld\n",
           params.total_elems, e->w.data - (precision_t*)params.mem,
           e->b.data - (precision_t*)params.mem, d->w.data - (precision_t*)params.mem,
           d->b.data - (precision_t*)params.mem, g->wi.data - (precision_t*)params.mem,
           g->wh.data - (precision_t*)params.mem, g->bi.data - (precision_t*)params.mem,
           g->bh.data - (precision_t*)params.mem, g->h0.data - (precision_t*)params.mem);
    if (emit) {
        // blob order, as policy.h lays it out; consumed by raptor_weights.py
        struct Slice { const char* name; PrecisionTensor* t; };
        Slice sl[] = {{"w0", &e->w}, {"b0", &e->b}, {"wi", &g->wi}, {"wh", &g->wh},
            {"bi", &g->bi}, {"bh", &g->bh}, {"h0", &g->h0}, {"w2", &d->w}, {"b2", &d->b},
            {"logstd", &d->logstd}};
        printf("total %ld\n", (long)(params.total_bytes / sizeof(precision_t)));
        for (Slice& s : sl)
            printf("%s %ld %ld\n", s.name, s.t->data - (precision_t*)params.mem,
                   (long)numel(s.t->shape));
    }
    return bad != 0;
}

// One Muon step must move every registered tensor at the pointer the allocator assigned.
static int optim_placement() {
    int TT = 8, B = 4;
    Encoder enc = {.in_dim = OBS, .out_dim = HID};
    create_raptor_encoder(&enc);
    Decoder dec = {.hidden_dim = HID, .output_dim = ACT, .continuous = true};
    create_raptor_decoder(&dec);
    Network net = {.hidden = HID, .num_layers = 1, .horizon = TT};
    create_raptor_network(&net);
    void* ew = enc.create_weights(&enc);
    void* dwp = dec.create_weights(&dec);
    void* nw = net.create_weights(&net);
    Allocator params = {}, mopt = {};
    enc.reg_params(ew, &params);
    dec.reg_params(dwp, &params);
    net.reg_params(nw, &params);
    alloc_create(&params);

    long span = params.total_bytes / sizeof(precision_t);
    PrecisionTensor grads = {.shape = {span}};
    cudaMalloc(&grads.data, span * sizeof(precision_t));
    std::vector<float> ones(span, 1.0f);
    cudaMemcpy(grads.data, ones.data(), span * sizeof(float), cudaMemcpyHostToDevice);
    FloatTensor w = {.data = (float*)params.mem, .shape = {span}};
    cudaMemset(params.mem, 0, span * sizeof(float));

    Muon m = {};
    muon_init(&m, &params, 1e-2, 0.9, 1e-8, 0.0, &mopt);
    alloc_create(&mopt);
    muon_post_create(&m);
    muon_step(&m, w, grads, 1.0f, 0);
    cudaDeviceSynchronize();

    std::vector<float> after(span);
    cudaMemcpy(after.data(), params.mem, span * sizeof(float), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < params.num_regs; i++) {
        AllocEntry& e = params.regs[i];
        long off = (precision_t*)(*e.data_ptr) - (precision_t*)params.mem;
        long ne = numel(e.shape), moved = 0;
        for (long k = 0; k < ne; k++) moved += after[off + k] != 0.0f;
        if (moved != ne) bad++;
    }
    printf("optimizer %d/%d registered tensors updated at their own offsets  %s\n",
           params.num_regs - bad, params.num_regs, bad ? "FAIL" : "PASS");
    return bad != 0;
}

// Decoder gradients, including the privileged critic, against central differences.
static int decoder_grad() {
    int TT = 4, B = 4, BT = B * TT;
    std::vector<float> pw = read_file("../../../../artifacts/raptor/base_policy.bin", 0);
    Decoder dec = {.hidden_dim = HID, .output_dim = ACT, .continuous = true};
    create_raptor_decoder(&dec);
    void* dwp = dec.create_weights(&dec);
    Allocator params = {}, acts = {}, grads = {};
    dec.reg_params(dwp, &params);
    void* dec_a = calloc(1, dec.activation_size);
    dec.reg_train(dwp, dec_a, &acts, &grads, BT);
    alloc_create(&params);
    alloc_create(&acts);
    alloc_create(&grads);

    RaptorDecWeights* d = (RaptorDecWeights*)dwp;
    RaptorDecActs* da = (RaptorDecActs*)dec_a;
    upload(&d->w, pw.data() + 2016);
    upload(&d->b, pw.data() + 2080);
    srand(11);
    std::vector<float> hid(BT * HID), obs(BT * OBS), gl(BT * ACT), gv(BT);
    for (float& v : hid) v = 2.0f * rand() / RAND_MAX - 1.0f;
    for (float& v : obs) v = 2.0f * rand() / RAND_MAX - 1.0f;
    for (float& v : gl) v = 2.0f * rand() / RAND_MAX - 1.0f;
    for (float& v : gv) v = 2.0f * rand() / RAND_MAX - 1.0f;
    std::vector<float> cw(numel(d->c1w.shape) + numel(d->c2w.shape) + numel(d->c3w.shape));
    for (float& v : cw) v = 0.2f * (2.0f * rand() / RAND_MAX - 1.0f);
    upload(&d->c1w, cw.data());
    upload(&d->c2w, cw.data() + numel(d->c1w.shape));
    upload(&d->c3w, cw.data() + numel(d->c1w.shape) + numel(d->c2w.shape));

    PrecisionTensor h = {.shape = {BT, HID}}, o = {.shape = {BT, OBS}};
    cudaMalloc(&h.data, hid.size() * sizeof(float));
    cudaMalloc(&o.data, obs.size() * sizeof(float));
    cudaMemcpy(h.data, hid.data(), hid.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(o.data, obs.data(), obs.size() * sizeof(float), cudaMemcpyHostToDevice);
    FloatTensor fl = {.shape = {BT, ACT}}, fv = {.shape = {BT}}, fs = {};
    cudaMalloc(&fl.data, gl.size() * sizeof(float));
    cudaMalloc(&fv.data, gv.size() * sizeof(float));
    cudaMemcpy(fl.data, gl.data(), gl.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fv.data, gv.data(), gv.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> host(BT * (ACT + 1));
    auto loss = [&]() {
        PrecisionTensor out = dec.forward(dwp, dec_a, h, o, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(host.data(), out.data, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double s = 0;
        for (int r = 0; r < BT; r++) {
            for (int c = 0; c < ACT; c++) s += (double)host[r * (ACT + 1) + c] * gl[r * ACT + c];
            s += (double)host[r * (ACT + 1) + ACT] * gv[r];
        }
        return s;
    };

    loss();
    dec.backward(dwp, dec_a, fl, fs, fv, 0);
    cudaDeviceSynchronize();

    struct Item { const char* name; PrecisionTensor* w; PrecisionTensor* grad; };
    Item items[] = {
        {"dec.w", &d->w, &da->wgrad}, {"dec.b", &d->b, &da->bgrad},
        {"critic.c1w", &d->c1w, &da->c1wgrad}, {"critic.c1b", &d->c1b, &da->c1bgrad},
        {"critic.c2w", &d->c2w, &da->c2wgrad}, {"critic.c3w", &d->c3w, &da->c3wgrad},
        {"critic.c3b", &d->c3b, &da->c3bgrad},
    };
    int fails = 0;
    float eps = 1e-3f;
    for (Item& it : items) {
        int total = numel(it.w->shape);
        std::vector<float> gr(total);
        cudaMemcpy(gr.data(), it.grad->data, total * sizeof(float), cudaMemcpyDeviceToHost);
        float worst = 0;
        for (int s = 0; s < 6; s++) {
            int i = (s * 7919 + 13) % total;
            float orig, up, dn;
            cudaMemcpy(&orig, it.w->data + i, sizeof(float), cudaMemcpyDeviceToHost);
            up = orig + eps;
            dn = orig - eps;
            cudaMemcpy(it.w->data + i, &up, sizeof(float), cudaMemcpyHostToDevice);
            double lp = loss();
            cudaMemcpy(it.w->data + i, &dn, sizeof(float), cudaMemcpyHostToDevice);
            double lm = loss();
            cudaMemcpy(it.w->data + i, &orig, sizeof(float), cudaMemcpyHostToDevice);
            float fd = (lp - lm) / (2 * eps);
            float rel = fabsf(fd - gr[i]) / fmaxf(1.0f, fmaxf(fabsf(fd), fabsf(gr[i])));
            if (rel > worst) worst = rel;
        }
        fails += worst > 2e-2f;
        printf("gradcheck %-11s max relative error = %.3e  %s\n", it.name, worst,
               worst <= 2e-2f ? "PASS" : "FAIL");
    }
    cudaMemset(fl.data, 0, gl.size() * sizeof(float));
    loss();
    PrecisionTensor dh = dec.backward(dwp, dec_a, fl, fs, fv, 0);
    cudaDeviceSynchronize();
    int leaked = 0;
    for (PrecisionTensor* t : {&dh, &da->wgrad, &da->bgrad}) {
        std::vector<float> grad(numel(t->shape));
        cudaMemcpy(grad.data(), t->data, grad.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (float v : grad) leaked += v != 0.0f;
    }
    std::vector<float> critic_grad(numel(da->c3wgrad.shape));
    cudaMemcpy(critic_grad.data(), da->c3wgrad.data, critic_grad.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    float critic_norm = 0;
    for (float v : critic_grad) critic_norm += v * v;
    bool detached = leaked == 0 && std::isfinite(critic_norm) && critic_norm > 0;
    printf("critic-only loss leaves actor/trunk gradients zero  %s\n", detached ? "PASS" : "FAIL");
    fails += !detached;
    return fails;
}

// The trainer seeds through init_weights + RAPTOR_POLICY_BLOB, not manual uploads.
static int blob_init() {
    setenv("RAPTOR_POLICY_BLOB", "../../../../artifacts/raptor/base_policy.bin", 1);
    std::vector<float> pw = read_file("../../../../artifacts/raptor/base_policy.bin", 0);
    Encoder enc = {.in_dim = OBS, .out_dim = HID};
    create_raptor_encoder(&enc);
    Decoder dec = {.hidden_dim = HID, .output_dim = ACT, .continuous = true};
    create_raptor_decoder(&dec);
    Network net = {.hidden = HID, .num_layers = 1, .horizon = 8};
    create_raptor_network(&net);
    void* ew = enc.create_weights(&enc);
    void* dwp = dec.create_weights(&dec);
    void* nw = net.create_weights(&net);
    Allocator params = {};
    enc.reg_params(ew, &params);
    dec.reg_params(dwp, &params);
    net.reg_params(nw, &params);
    alloc_create(&params);
    uint64_t seed = 7;
    enc.init_weights(ew, &seed, 0);
    dec.init_weights(dwp, &seed, 0);
    net.init_weights(nw, &seed, 0);
    cudaDeviceSynchronize();

    RaptorEncWeights* e = (RaptorEncWeights*)ew;
    RaptorDecWeights* d = (RaptorDecWeights*)dwp;
    RaptorGRUWeights* g = (RaptorGRUWeights*)nw;
    struct Slot { const char* name; PrecisionTensor* t; long offset; };
    Slot slots[] = {{"enc.w", &e->w, 0}, {"enc.b", &e->b, 352}, {"gru.wi", &g->wi, 368},
        {"gru.wh", &g->wh, 1136}, {"gru.bi", &g->bi, 1904}, {"gru.bh", &g->bh, 1952},
        {"gru.h0", &g->h0, 2000}, {"actor.w", &d->w, 2016}, {"actor.b", &d->b, 2080}};
    int bad = 0;
    long checked = 0;
    for (Slot& s : slots) {
        long n = numel(s.t->shape);
        std::vector<float> got(n);
        cudaMemcpy(got.data(), s.t->data, n * sizeof(float), cudaMemcpyDeviceToHost);
        for (long i = 0; i < n; i++) bad += got[i] != pw[s.offset + i];
        checked += n;
    }
    printf("blob init %ld/%ld actor floats seeded bit-exactly from the shipped policy  %s\n",
           checked - bad, checked, bad ? "FAIL" : "PASS");
    return bad != 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--layout") == 0) return layout(true);
    int fails = parity() + parity_rollout() + gradcheck(false) + gradcheck(true) + layout()
        + optim_placement() + decoder_grad() + blob_init();
    printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails != 0;
}
