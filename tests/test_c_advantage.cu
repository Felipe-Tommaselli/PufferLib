#include "../src/kernels.cu"
#include <vector>

static precision_t* upload(const std::vector<float>& values) {
    std::vector<precision_t> host(values.size());
    for (size_t i = 0; i < values.size(); i++) host[i] = from_float(values[i]);
    precision_t* device;
    cudaMalloc(&device, host.size() * sizeof(precision_t));
    cudaMemcpy(device, host.data(), host.size() * sizeof(precision_t), cudaMemcpyHostToDevice);
    return device;
}

static int check(const char* name, int T, const std::vector<float>& values,
        const std::vector<float>& rewards, const std::vector<float>& dones,
        const std::vector<float>& importance, const std::vector<float>& timeouts,
        const std::vector<float>& expected, bool with_tail) {
    const int B = 3;
    precision_t* v = upload(values), *r = upload(rewards), *d = upload(dones);
    precision_t* imp = upload(importance), *tv = timeouts.empty() ? nullptr : upload(timeouts);
    precision_t* advantages = upload(std::vector<float>(B * T, -555.0f));
    precision_t* tail = with_tail ? upload(std::vector<float>(3 * B)) : nullptr;
    precision_t* output = upload({-77.0f, 4.0f, -77.0f, 6.0f, -77.0f, 99.0f});
    float env_host[] = {0.25f, 0.25f, 0.25f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    float* env;
    cudaMalloc(&env, sizeof(env_host));
    cudaMemcpy(env, env_host, sizeof(env_host), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int failures = 0;
    constexpr int N = 16 / sizeof(precision_t);
    for (int vectorized = 0; vectorized <= (T % N == 0); vectorized++) {
        for (int captured = 0; captured < 2; captured++) {
            if (captured) cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            if (tail) store_bootstrap_tail<<<1, 32, 0, stream>>>(
                tail, output, env, env + B, env + 2 * B, 0, B, B, 2);
            auto kernel = vectorized ? puff_advantage : puff_advantage_scalar;
            kernel<<<1, 32, 0, stream>>>(v, r, d, imp, advantages,
                0.5f, 0.5f, 1.0f, 1.0f, B, T, tv, tail);
            cudaGraphExec_t executable = nullptr;
            if (captured) {
                cudaGraph_t graph;
                cudaStreamEndCapture(stream, &graph);
                cudaGraphInstantiate(&executable, graph, 0);
                cudaGraphDestroy(graph);
                cudaGraphLaunch(executable, stream);
            }
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                printf("%s: %s\n", name, cudaGetErrorString(err));
                exit(1);
            }
            std::vector<precision_t> host(B * T);
            cudaMemcpy(host.data(), advantages, host.size() * sizeof(precision_t), cudaMemcpyDeviceToHost);
            int bad = 0;
            for (int i = 0; i < B * T; i++) {
                float got = to_float(host[i]);
                float tolerance = USE_BF16 ? 0.04f : 1e-5f;
                if (!isfinite(got) || fabsf(got - expected[i]) > tolerance) {
                    printf("%s row=%d t=%d: got %g expected %g\n", name, i / T, i % T, got, expected[i]);
                    bad++;
                }
            }
            failures += bad;
            printf("%s %s %s: %s\n", name, vectorized ? "vector" : "scalar",
                captured ? "graph" : "eager", bad ? "FAIL" : "PASS");
            if (executable) cudaGraphExecDestroy(executable);
        }
    }
    cudaStreamDestroy(stream);
    cudaFree(v); cudaFree(r); cudaFree(d); cudaFree(imp); cudaFree(tv);
    cudaFree(advantages); cudaFree(tail); cudaFree(output); cudaFree(env);
    return failures;
}

int main() {
    int failures = check("horizon-one", 1, {2, 2, 2}, {777, 777, 777}, {1, 1, 1},
        {0.5f, 1, 2}, {}, {0.125f, 1.25f, -1.75f}, true);
    failures += check("odd-horizon-boundaries", 3,
        {2, 10, 14, 2, 10, 14, 2, 10, 14},
        {777, 0.25f, 0.25f, 777, 0.25f, 0.25f, 777, 0.25f, 0.25f},
        {0, 1, 1, 0, 1, 1, 0, 1, 1}, std::vector<float>(9, 0.5f),
        {0, 6, 0, 0, 6, 0, 0, 6, 0},
        {0.625f, -4.875f, -5.875f, 0.625f, -4.875f, -5.375f, 0.625f, -4.875f, -6.875f}, true);
    failures += check("generic-no-tail", 3, std::vector<float>(9, 2),
        std::vector<float>(9, 0.25f), {0, 0, 1, 0, 0, 1, 0, 0, 1},
        std::vector<float>(9, 0.5f), {},
        {-0.484375f, -0.875f, 0, -0.484375f, -0.875f, 0, -0.484375f, -0.875f, 0}, false);
    for (int T : {8, 16}) {
        std::vector<float> v(3 * T, 2), r(3 * T, 0.25f), d(3 * T), tv(3 * T), expected(3 * T);
        const float last[] = {0.25f, 1.25f, -1.75f};
        for (int b = 0; b < 3; b++) {
            for (int offset = 0; offset < T; offset += 8) {
                int i = b * T + offset;
                v[i + 3] = 10;
                v[i + 5] = 14;
                d[i + 3] = d[i + 5] = 1;
                tv[i + 3] = 6;
                float end = offset + 8 == T ? last[b] : 0.25f;
                const float want[] = {-0.859375f, -0.4375f, 1.25f, -9.1875f, -1.75f,
                    -12.75f + 0.25f * (-0.75f + 0.25f * end), -0.75f + 0.25f * end, end};
                for (int t = 0; t < 8; t++) expected[i + t] = want[t];
                if (offset) { d[i] = 1; tv[i] = 4; }
            }
        }
        failures += check(T == 8 ? "mixed-boundaries" : "vector-chunk-boundary", T,
            v, r, d, std::vector<float>(3 * T, 1), tv, expected, true);
    }
    return failures != 0;
}
