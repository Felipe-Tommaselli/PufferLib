#include <stdio.h>
#include <time.h>
#include "raptor.h"
#include "policy.h"

static int failures = 0;

static void check(const char* name, bool ok, const char* detail) {
    printf("%-34s %s  %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) failures++;
}

static float g_dr = 0.0f;
static TrajParams g_traj = {0.0f, 10.0f};

static RaptorEnv* make_env(int num_agents, unsigned int seed) {
    RaptorEnv* env = (RaptorEnv*)calloc(1, sizeof(RaptorEnv));
    env->num_agents = num_agents;
    env->rng = seed;
    env->airframe = AIRFRAME_X500;
    env->reward_params = REWARD_FOUNDATION;
    env->term_params = TERMINATION_FOUNDATION;
    env->init_params = INIT_90_DEG;
    env->horizon = RAPTOR_HORIZON;
    env->dr = g_dr;
    env->traj = g_traj;
    env->observations = (float*)calloc(num_agents * RAPTOR_OBS_DIM, sizeof(float));
    env->actions = (float*)calloc(num_agents * RAPTOR_ACT_DIM, sizeof(float));
    env->rewards = (float*)calloc(num_agents, sizeof(float));
    env->terminals = (float*)calloc(num_agents, sizeof(float));
    init(env);
    return env;
}

static void free_env(RaptorEnv* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
}

static void test_inertia(void) {
    const Airframe* p = &AIRFRAME_X500;
    float worst = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float v = 0;
            for (int k = 0; k < 3; k++) v += p->J[i][k] * p->J_inv[k][j];
            float target = (i == j) ? 1.0f : 0.0f;
            float e = fabsf(v - target);
            if (e > worst) worst = e;
        }
    char buf[64];
    snprintf(buf, sizeof(buf), "max|J*Jinv - I| = %.3e", worst);
    check("T3 inertia consistency", worst < 1e-5f, buf);
}

static void test_hover_equilibrium(void) {
    RaptorEnv* env = make_env(1, 1);
    Agent* agent = &env->agents[0];
    memset(&agent->state, 0, sizeof(State));
    agent->airframe = AIRFRAME_X500;
    agent->state.orientation[0] = 1;
    float hover = AIRFRAME_X500.hovering_throttle_relative;
    for (int i = 0; i < 4; i++) {
        agent->state.rpm[i] = hover;
        agent->state.last_action[i] = 2 * hover - 1;
    }
    for (int i = 0; i < RAPTOR_ACT_DIM; i++) env->actions[i] = 2 * hover - 1;

    float first_reward = 0;
    for (int t = 0; t < RAPTOR_HORIZON - 1; t++) {
        c_step(env);
        if (t == 0) first_reward = env->rewards[0];
    }
    float drift = sqrtf(agent->state.position[0] * agent->state.position[0] +
                        agent->state.position[1] * agent->state.position[1] +
                        agent->state.position[2] * agent->state.position[2]);
    char buf[96];
    snprintf(buf, sizeof(buf), "drift %.3e m over 500 steps, r0 = %.4f", drift, first_reward);
    check("T2 hover equilibrium", drift < 1e-3f && fabsf(first_reward - 1.5f) < 1e-3f, buf);
    free_env(env);
}

static void test_motor_step_response(void) {
    Airframe p = AIRFRAME_X500;
    State s;
    memset(&s, 0, sizeof(State));
    s.orientation[0] = 1;
    float action[4] = {1, 1, 1, 1};
    float tau = p.rotor_time_constants_rising[0];
    float worst = 0;
    for (int t = 1; t <= 20; t++) {
        State next;
        physics_step(&p, &s, action, &next);
        s = next;
        float expected = 1.0f - expf(-(t * RAPTOR_DT) / tau);
        float e = fabsf(s.rpm[0] - expected);
        if (e > worst) worst = e;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "max|rpm - (1-exp(-t/tau))| = %.3e", worst);
    check("T6 motor step response", worst < 1e-3f, buf);
}

static void test_termination_coverage(void) {
    RaptorEnv* env = make_env(1, 3);
    State s;
    memset(&s, 0, sizeof(State));
    s.orientation[0] = 1;
    bool none = !terminated(&env->term_params, &s);
    s.position[0] = 1.01f;
    bool pos = terminated(&env->term_params, &s);
    s.position[0] = 0;
    s.linear_velocity[1] = 2.01f;
    bool vel = terminated(&env->term_params, &s);
    s.linear_velocity[1] = 0;
    s.angular_velocity[2] = 35.01f;
    bool omega = terminated(&env->term_params, &s);
    s.angular_velocity[2] = 0;
    s.position[2] = NAN;
    bool nan_caught = terminated(&env->term_params, &s);
    check("T7 termination coverage", none && pos && vel && omega && nan_caught,
          "position, velocity, omega and NaN all fire");
    free_env(env);
}

static void test_autoreset(void) {
    RaptorEnv* env = make_env(64, 4);
    c_reset(env);
    int resets = 0;
    bool ok = true;
    for (int t = 0; t < 2000; t++) {
        for (int i = 0; i < 64 * RAPTOR_ACT_DIM; i++) env->actions[i] = rnd_uniform(&env->rng, -1.0f, 1.0f);
        c_step(env);
        for (int i = 0; i < 64; i++) {
            if (env->terminals[i] == 0.0f) continue;
            resets++;
            float* o = env->observations + i * RAPTOR_OBS_DIM;
            for (int k = 0; k < 3; k++)
                if (fabsf(o[k]) > env->init_params.max_position + 1e-6f) ok = false;
            for (int k = 12; k < 15; k++)
                if (fabsf(o[k]) > env->init_params.max_linear_velocity + 1e-6f) ok = false;
            for (int k = 18; k < 22; k++)
                if (o[k] != 0.0f) ok = false;
        }
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%d resets, all post-terminal obs in support", resets);
    check("T8 autoreset correctness", ok && resets > 0, buf);
    free_env(env);
}

static void test_reset_distribution(void) {
    RaptorEnv* env = make_env(4096, 5);
    c_reset(env);
    float mean[3] = {0, 0, 0};
    float max_abs = 0;
    int zero_position = 0;
    for (int i = 0; i < 4096; i++) {
        State* s = &env->agents[i].state;
        float n = 0;
        for (int k = 0; k < 3; k++) {
            mean[k] += s->position[k] / 4096.0f;
            if (fabsf(s->position[k]) > max_abs) max_abs = fabsf(s->position[k]);
            n += fabsf(s->position[k]);
        }
        if (n == 0.0f) zero_position++;
    }
    float guidance_frac = (float)zero_position / 4096.0f;
    bool centered = fabsf(mean[0]) < 0.03f && fabsf(mean[1]) < 0.03f && fabsf(mean[2]) < 0.03f;
    bool bounded = max_abs <= env->init_params.max_position + 1e-6f;
    bool guided = fabsf(guidance_frac - env->init_params.guidance) < 0.02f;
    char buf[96];
    snprintf(buf, sizeof(buf), "mean %.4f %.4f %.4f, max %.3f, guidance %.3f", mean[0], mean[1], mean[2],
             max_abs, guidance_frac);
    check("T9 reset distribution", centered && bounded && guided, buf);
    free_env(env);
}

static void test_buffer_contract(void) {
    RaptorEnv* env = make_env(4096, 6);
    c_reset(env);
    bool finite = true;
    for (int t = 0; t < 1000; t++) {
        for (int i = 0; i < 4096 * RAPTOR_ACT_DIM; i++) env->actions[i] = rnd_uniform(&env->rng, -1.0f, 1.0f);
        c_step(env);
        for (int i = 0; i < 4096 * RAPTOR_OBS_DIM; i++)
            if (!isfinite(env->observations[i])) finite = false;
        for (int i = 0; i < 4096; i++)
            if (!isfinite(env->rewards[i]) || !isfinite(env->terminals[i])) finite = false;
    }
    check("T1 buffer contract", finite, "4096 agents x 1000 steps, all finite");
    free_env(env);
}

static void test_determinism(void) {
    RaptorEnv* a = make_env(256, 7);
    RaptorEnv* b = make_env(256, 7);
    c_reset(a);
    c_reset(b);
    bool same = true;
    for (int t = 0; t < 200; t++) {
        for (int i = 0; i < 256 * RAPTOR_ACT_DIM; i++) {
            float v = sinf(0.01f * (t * 1024 + i));
            a->actions[i] = v;
            b->actions[i] = v;
        }
        c_step(a);
        c_step(b);
        for (int i = 0; i < 256 * RAPTOR_OBS_DIM; i++)
            if (a->observations[i] != b->observations[i]) same = false;
    }
    check("T12 determinism", same, "same seed, bit-identical observations");
    free_env(a);
    free_env(b);
}

static void test_throughput(void) {
    RaptorEnv* env = make_env(4096, 8);
    c_reset(env);
    clock_t start = clock();
    int steps = 500;
    for (int t = 0; t < steps; t++) {
        for (int i = 0; i < 4096 * RAPTOR_ACT_DIM; i++) env->actions[i] = 0.1f;
        c_step(env);
    }
    double secs = (double)(clock() - start) / CLOCKS_PER_SEC;
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f M agent-steps/s (single thread)", 4096.0 * steps / secs / 1e6);
    check("T13 throughput", secs > 0, buf);
    free_env(env);
}

static void test_yaw_sign(void) {
    Airframe p = AIRFRAME_X500;
    State s;
    memset(&s, 0, sizeof(State));
    s.orientation[0] = 1;
    float hover = p.hovering_throttle_relative;
    for (int i = 0; i < 4; i++) s.rpm[i] = hover;
    float action[4] = {2 * hover - 1, 1.0f, 2 * hover - 1, 1.0f};
    State next;
    physics_step(&p, &s, action, &next);
    char buf[72];
    snprintf(buf, sizeof(buf), "CCW pair (1,3) up -> omega_z = %+.4f", next.angular_velocity[2]);
    check("T5 yaw torque sign", next.angular_velocity[2] > 0, buf);
}

static void test_policy_parity(void) {
    float worst = 0;
    for (int b = 0; b < EX_BATCH; b++) {
        PolicyState st;
        policy_reset(&st);
        for (int t = 0; t < EX_STEPS; t++) {
            float a[RAPTOR_ACT_DIM];
            policy_step(&st, &EX_IN[(t * EX_BATCH + b) * RAPTOR_OBS_DIM], a);
            for (int k = 0; k < RAPTOR_ACT_DIM; k++) {
                float e = fabsf(a[k] - EX_OUT[(t * EX_BATCH + b) * RAPTOR_ACT_DIM + k]);
                if (e > worst) worst = e;
            }
        }
    }
    char buf[80];
    snprintf(buf, sizeof(buf), "max|C - shipped example| = %.3e over %d steps", worst, EX_STEPS);
    check("T10 policy forward parity", worst < 1e-5f, buf);
}

typedef struct {
    int episodes;
    float ret, length, position_error, d_action, terminated;
} Replay;

static Replay run_replay(int n, int steps) {
    RaptorEnv* env = make_env(n, 11);
    PolicyState* policies = (PolicyState*)calloc(n, sizeof(PolicyState));
    c_reset(env);
    for (int i = 0; i < n; i++) policy_reset(&policies[i]);
    for (int t = 0; t < steps; t++) {
        for (int i = 0; i < n; i++)
            policy_step(&policies[i], env->observations + i * RAPTOR_OBS_DIM,
                        env->actions + i * RAPTOR_ACT_DIM);
        c_step(env);
        for (int i = 0; i < n; i++)
            if (env->terminals[i] != 0.0f) policy_reset(&policies[i]);
    }
    float e = env->log.n > 0 ? env->log.n : 1.0f;
    Replay r = {(int)env->log.n,          env->log.episode_return / e, env->log.episode_length / e,
                env->log.position_error / e, env->log.d_action / e,    env->log.terminated / e};
    free(policies);
    free_env(env);
    return r;
}

static void test_base_policy_replay(void) {
    Replay r = run_replay(256, 5000);
    printf("     episodes %d | return %.2f | length %.1f | pos err %.3f m | terminated %.1f%%\n",
           r.episodes, r.ret, r.length, r.position_error, 100.0f * r.terminated);
    char buf[96];
    snprintf(buf, sizeof(buf), "return %.2f, length %.1f, terminated %.1f%%", r.ret, r.length,
             100.0f * r.terminated);
    check("T11 base-policy closed-loop replay", r.terminated < 0.05f && r.length > 450.0f, buf);
}

static void dump_trajectories(int episodes) {
    RaptorEnv* env = make_env(episodes, 21);
    PolicyState* policies = (PolicyState*)calloc(episodes, sizeof(PolicyState));
    State* pre = (State*)calloc(episodes, sizeof(State));
    c_reset(env);
    for (int i = 0; i < episodes; i++) policy_reset(&policies[i]);
    printf("episode,step,x,y,z,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,dist,reward,a0,a1,a2,a3,terminal,tx,ty,tz\n");
    for (int t = 0; t < RAPTOR_HORIZON; t++) {
        for (int i = 0; i < episodes; i++) {
            policy_step(&policies[i], env->observations + i * RAPTOR_OBS_DIM,
                        env->actions + i * RAPTOR_ACT_DIM);
            pre[i] = env->agents[i].state;
        }
        c_step(env);
        for (int i = 0; i < episodes; i++) {
            State* s = &pre[i];
            float* a = env->actions + i * RAPTOR_ACT_DIM;
            float ex = s->position[0] - s->target[0], ey = s->position[1] - s->target[1],
                  ez = s->position[2] - s->target[2];
            float d = sqrtf(ex * ex + ey * ey + ez * ez);
            printf("%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.0f,%.5f,%.5f,%.5f\n",
                   i, t, s->position[0], s->position[1], s->position[2], s->linear_velocity[0],
                   s->linear_velocity[1], s->linear_velocity[2], s->angular_velocity[0],
                   s->angular_velocity[1], s->angular_velocity[2], s->orientation[0],
                   s->orientation[1], s->orientation[2], s->orientation[3], d,
                   env->rewards[i], a[0], a[1], a[2], a[3], env->terminals[i],
                   s->target[0], s->target[1], s->target[2]);
            if (env->terminals[i] != 0.0f) policy_reset(&policies[i]);
        }
    }
    free(pre);
    free(policies);
    free_env(env);
}

static void write_seq(const char* path, int T, int B, int dim, const float* data) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(1);
    }
    int header[2] = {T, B};
    fwrite(header, sizeof(int), 2, f);
    fwrite(data, sizeof(float), (size_t)T * B * dim, f);
    fclose(f);
}

static void policy_eval_file(const char* in_path, const char* out_path) {
    FILE* f = fopen(in_path, "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", in_path);
        exit(1);
    }
    int header[2];
    if (fread(header, sizeof(int), 2, f) != 2) exit(1);
    int T = header[0], B = header[1];
    float* in = (float*)malloc((size_t)T * B * RAPTOR_OBS_DIM * sizeof(float));
    float* out = (float*)malloc((size_t)T * B * RAPTOR_ACT_DIM * sizeof(float));
    if (fread(in, sizeof(float), (size_t)T * B * RAPTOR_OBS_DIM, f) != (size_t)T * B * RAPTOR_OBS_DIM)
        exit(1);
    fclose(f);
    for (int b = 0; b < B; b++) {
        PolicyState st;
        policy_reset(&st);
        for (int t = 0; t < T; t++)
            policy_step(&st, &in[((size_t)t * B + b) * RAPTOR_OBS_DIM],
                        &out[((size_t)t * B + b) * RAPTOR_ACT_DIM]);
    }
    write_seq(out_path, T, B, RAPTOR_ACT_DIM, out);
    free(in);
    free(out);
}

int main(int argc, char** argv) {
    policy_init();
    const char* mode = NULL;
    const char* mode_args[3] = {NULL, NULL, NULL};
    int n_mode_args = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            policy_load(argv[++i]);
        } else if (strcmp(argv[i], "--dr") == 0 && i + 1 < argc) {
            g_dr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--fig8") == 0 && i + 1 < argc) {
            g_traj.amplitude = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--fig8-period") == 0 && i + 1 < argc) {
            g_traj.period = (float)atof(argv[++i]);
        } else if (strncmp(argv[i], "--", 2) == 0) {
            mode = argv[i];
            n_mode_args = 0;
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0 && n_mode_args < 3)
                mode_args[n_mode_args++] = argv[++i];
        }
    }

    if (mode && strcmp(mode, "--export-weights") == 0) {
        policy_save(mode_args[0]);
        return 0;
    }
    if (mode && strcmp(mode, "--fixture-out") == 0) {
        char path[512];
        snprintf(path, sizeof(path), "%s_in.bin", mode_args[0]);
        write_seq(path, EX_STEPS, EX_BATCH, RAPTOR_OBS_DIM, EX_IN);
        snprintf(path, sizeof(path), "%s_out.bin", mode_args[0]);
        write_seq(path, EX_STEPS, EX_BATCH, RAPTOR_ACT_DIM, EX_OUT);
        return 0;
    }
    if (mode && strcmp(mode, "--policy-eval") == 0) {
        policy_eval_file(mode_args[0], mode_args[1]);
        return 0;
    }
    if (mode && strcmp(mode, "--replay") == 0) {
        Replay r = run_replay(mode_args[0] ? atoi(mode_args[0]) : 256,
                              mode_args[1] ? atoi(mode_args[1]) : 5000);
        printf("{\"episodes\": %d, \"return\": %.6f, \"length\": %.4f, \"position_error\": %.6f, "
               "\"d_action\": %.6f, \"terminated\": %.6f}\n",
               r.episodes, r.ret, r.length, r.position_error, r.d_action, r.terminated);
        return 0;
    }
    if (mode && strcmp(mode, "--dump") == 0) {
        dump_trajectories(mode_args[0] ? atoi(mode_args[0]) : 8);
        return 0;
    }

    printf("\nRAPTOR ocean environment - Phase 0 (no learning)\n\n");
    test_buffer_contract();
    test_hover_equilibrium();
    test_inertia();
    test_yaw_sign();
    test_motor_step_response();
    test_termination_coverage();
    test_autoreset();
    test_reset_distribution();
    test_determinism();
    test_policy_parity();
    test_base_policy_replay();
    test_throughput();
    printf("\n%s (%d failures)\n\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures != 0;
}
