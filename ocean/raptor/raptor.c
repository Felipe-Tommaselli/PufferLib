#include <stdio.h>
#include <time.h>
#include "raptor.h"
#include "policy.h"

static int failures = 0;

static void check(const char* name, bool ok, const char* detail) {
    printf("%-34s %s  %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) failures++;
}

static RaptorEnv g_settings;
static int g_airframe = 0;
static unsigned int g_seed = 11;

static RaptorEnv* make_env(int num_agents, unsigned int seed) {
    RaptorEnv* env = (RaptorEnv*)calloc(1, sizeof(RaptorEnv));
    env->num_agents = num_agents;
    env->rng = seed;
    env->airframe = AIRFRAMES[g_airframe];
    env->reward_params = g_settings.reward_params;
    env->term_params = g_settings.term_params;
    env->init_params = g_settings.init_params;
    env->horizon = g_settings.horizon;
    env->obs_clip = g_settings.obs_clip;
    env->dr = g_settings.dr;
    env->traj = g_settings.traj;
    env->yaw_face_probability = g_settings.yaw_face_probability;
    env->yaw_rate = g_settings.yaw_rate;
    env->observations = (float*)calloc(num_agents * RAPTOR_OBS_DIM, sizeof(float));
    env->actions = (float*)calloc(num_agents * RAPTOR_ACT_DIM, sizeof(float));
    env->rewards = (float*)calloc(num_agents, sizeof(float));
    env->terminals = (float*)calloc(num_agents, sizeof(float));
    env->truncations = (float*)calloc(num_agents, sizeof(float));
    env->final_observations = (float*)calloc(num_agents * RAPTOR_OBS_DIM, sizeof(float));
    init(env);
    return env;
}

static void free_env(RaptorEnv* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncations);
    free(env->final_observations);
    free(env);
}

static void test_inertia(void) {
    const Airframe* p = &AIRFRAME_IMAV;
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
    agent->airframe = AIRFRAME_IMAV;
    agent->state.orientation[0] = 1;
    float hover = AIRFRAME_IMAV.hovering_throttle_relative;
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
    Airframe p = AIRFRAME_IMAV;
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
            float error = s->position[k] - s->target[k];
            float c = cosf(s->target_yaw), sn = sinf(s->target_yaw);
            if (k == 0) error = c * (s->position[0] - s->target[0]) + sn * (s->position[1] - s->target[1]);
            if (k == 1) error = -sn * (s->position[0] - s->target[0]) + c * (s->position[1] - s->target[1]);
            mean[k] += error / 4096.0f;
            if (fabsf(error) > max_abs) max_abs = fabsf(error);
            n += fabsf(error);
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
    Airframe p = AIRFRAME_IMAV;
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

static void test_native_contract(void) {
    State s = {0};
    s.orientation[0] = 1;
    s.position[0] = 0.2f;
    s.position[1] = 0.2f;
    s.linear_velocity[0] = 2;
    s.last_action[0] = 1.7f;
    float obs[22], rotated[22], action[4] = {0};
    observe(&s, obs, true);
    RewardParams r = REWARD_FOUNDATION;
    r.orientation = r.d_action = 0;
    bool cap = obs[0] == 0.1f && obs[1] == 0.1f && obs[12] == 1 &&
               obs[18] == 1.7f && fabsf(reward(&AIRFRAME_IMAV, &r, &s, action, &s, false) - 1.4f) < 1e-6f;
    check("T14 native clips", cap, "per-axis observations, radial reward, raw action history");
    s.position[0] = 0.04f;
    s.position[1] = -0.02f;
    s.linear_velocity[0] = 0.3f;
    s.linear_velocity[1] = -0.4f;
    s.orientation[0] = cosf(0.2f);
    s.orientation[3] = sinf(0.2f);
    observe(&s, obs, true);
    r.orientation = 0.1f;
    float before = reward(&AIRFRAME_IMAV, &r, &s, action, &s, false);
    State turned = s;
    turned.target_yaw = 1.57079632679f;
    turned.position[0] = -s.position[1];
    turned.position[1] = s.position[0];
    turned.linear_velocity[0] = -s.linear_velocity[1];
    turned.linear_velocity[1] = s.linear_velocity[0];
    turned.orientation[0] = cosf(0.2f + 0.78539816339f);
    turned.orientation[3] = sinf(0.2f + 0.78539816339f);
    observe(&turned, rotated, true);
    bool equivariant = true;
    for (int k = 0; k < 22; k++) equivariant &= fabsf(obs[k] - rotated[k]) < 1e-6f;
    equivariant &= fabsf(before - reward(&AIRFRAME_IMAV, &r, &turned, action, &turned, false)) < 1e-6f;
    check("T15 target yaw equivariance", equivariant, "common world yaw preserves observation and inherited reward");

    unsigned int seed = 19;
    bool bounds = true;
    for (int n = 0; n < 2048; n++) {
        Airframe p = AIRFRAME_IMAV;
        randomize_airframe(&p, 1, &seed);
        float factors[] = {p.mass / AIRFRAME_IMAV.mass,
            p.rotor_thrust_coefficients[0][2] / AIRFRAME_IMAV.rotor_thrust_coefficients[0][2],
            p.J[0][0] / AIRFRAME_IMAV.J[0][0], p.J[1][1] / AIRFRAME_IMAV.J[1][1],
            p.J[2][2] / AIRFRAME_IMAV.J[2][2],
            p.rotor_torque_constants[0] / AIRFRAME_IMAV.rotor_torque_constants[0],
            p.rotor_time_constants_rising[0] / AIRFRAME_IMAV.rotor_time_constants_rising[0],
            p.rotor_time_constants_falling[0] / AIRFRAME_IMAV.rotor_time_constants_falling[0],
            p.rotor_positions[0][0] / AIRFRAME_IMAV.rotor_positions[0][0]};
        for (int k = 0; k < 9; k++) {
            float width = k < 2 ? 0.15f : 0.04f;
            bounds &= factors[k] >= 1-width-1e-6f && factors[k] <= 1+width+1e-6f;
        }
        bounds &= p.hovering_throttle_relative > 0 && p.hovering_throttle_relative < 1;
        for (int k = 0; k < 3; k++) bounds &= fabsf(p.J[k][k] * p.J_inv[k][k] - 1) < 1e-6f;
    }
    Airframe nominal = AIRFRAME_IMAV;
    randomize_airframe(&nominal, 0, &seed);
    bounds &= memcmp(&nominal, &AIRFRAME_IMAV, sizeof(nominal)) == 0;
    check("T16 DR envelope", bounds, "mass/thrust 15%, others 4%, positive plant and nominal dr=0");
}

static void test_command_continuity(void) {
    bool coherent = true;
    for (int circuit = 0; circuit < 2; circuit++) {
        Trajectory tr = {.amplitude=.5f, .period=15, .phase=.7f, .direction=-1,
                         .heading=.4f, .ramp=2, .circuit=circuit, .aspect=.5f};
        for (int n = 1; n < 2000; n++) {
            float t = n * .01f, p[3], v[3], lo[3], hi[3], vl[3], vh[3];
            trajectory(&tr, t, p, v);
            trajectory(&tr, t-.001f, lo, vl);
            trajectory(&tr, t+.001f, hi, vh);
            float speed = 0, acceleration = 0;
            for (int k = 0; k < 3; k++) {
                coherent &= fabsf((hi[k]-lo[k])/.002f-v[k]) < .001f;
                speed += v[k]*v[k];
                acceleration += (vh[k]-vl[k])*(vh[k]-vl[k])/.000004f;
            }
            coherent &= speed <= .3f*.3f && acceleration <= .5f*.5f;
        }
        tr.ramp = 0;
        float p0[3], v0[3], p1[3], v1[3];
        trajectory(&tr, 0, p0, v0);
        trajectory(&tr, tr.period, p1, v1);
        for (int k = 0; k < 3; k++)
            coherent &= fabsf(p0[k]-p1[k]) < 1e-6f && fabsf(v0[k]-v1[k]) < 1e-6f;
    }
    RaptorEnv* env = make_env(64, 123);
    env->init_params.guidance = 1;
    env->yaw_face_probability = 1;
    c_reset(env);
    float yaw[64];
    for (int i = 0; i < 64; i++) {
        State* s = &env->agents[i].state;
        for (int k = 0; k < 3; k++)
            coherent &= s->position[k] == s->target[k] && s->linear_velocity[k] == s->target_velocity[k];
        yaw[i] = s->target_yaw;
        for (int k = 0; k < 4; k++) env->actions[4*i+k] = 2*env->agents[i].airframe.hovering_throttle_relative-1;
    }
    c_step(env);
    for (int i = 0; i < 64; i++) {
        float delta = env->agents[i].state.target_yaw - yaw[i];
        coherent &= fabsf(delta) <= env->yaw_rate * RAPTOR_DT + 1e-6f;
    }
    env->yaw_face_probability = 0;
    c_reset(env);
    for (int i = 0; i < 64; i++) coherent &= env->agents[i].state.target_yaw == 0;
    free_env(env);
    env = make_env(4096, 124);
    env->traj.original = 1;
    env->traj.moving_fraction = 0.5f;
    c_reset(env);
    int moving = 0, vertical = 0;
    float min_yaw = 4, max_yaw = -4;
    for (int i = 0; i < 4096; i++) {
        Trajectory* tr = &env->agents[i].trajectory;
        moving += tr->moving;
        original_trajectory(tr, env->agents[i].state.target,
                            env->agents[i].state.target_velocity, &env->rng);
        vertical += fabsf(tr->velocity[2]) > 0;
        float yaw = env->agents[i].state.target_yaw;
        if (yaw < min_yaw) min_yaw = yaw;
        if (yaw > max_yaw) max_yaw = yaw;
    }
    coherent &= moving > 1900 && moving < 2200 && vertical == moving &&
                min_yaw < -3 && max_yaw > 3;
    check("T17 continuous task commands", coherent,
          "periodic evaluation; original 50/50 3D commands with full independent yaw");
    free_env(env);
}

static void test_timeout_and_applied_action(void) {
    RaptorEnv* env = make_env(1, 234);
    env->dr = 0;
    env->horizon = 1;
    env->traj.moving_fraction = 0;
    env->init_params.guidance = 1;
    env->yaw_face_probability = 0;
    c_reset(env);
    env->agents[0].state.position[0] = .03f;
    for (int k = 0; k < 4; k++) {
        env->agents[0].state.last_action[k] = 2;
        env->actions[k] = 3;
    }
    c_step(env);
    bool timeout = env->terminals[0] == 1 && env->truncations[0] == 1 &&
        env->final_observations[18] == 3 && env->observations[18] == 0 &&
        fabsf(env->final_observations[0]) > .02f &&
        env->log.d_action == 2 && env->log.applied_d_action == 0 && env->log.saturation == 1;
    env->agents[0].state.position[0] = 2;
    c_step(env);
    bool crash = env->terminals[0] == 1 && env->truncations[0] == 0 && env->log.terminated == 1;
    check("T18 timeout and applied command", timeout && crash,
          "final observation retained, raw history preserved, clipping measured, crash beats timeout");
    env->horizon = 8;
    env->yaw_face_probability = 1;
    c_reset(env);
    memset(&env->log, 0, sizeof(env->log));
    State* s = &env->agents[0].state;
    float c = cosf(s->target_yaw), sn = sinf(s->target_yaw);
    s->position[0] = c*.03f + sn*.02f;
    s->position[1] = sn*.03f - c*.02f;
    s->position[2] = .01f;
    float hover = env->agents[0].airframe.hovering_throttle_relative;
    for (int k = 0; k < 4; k++) {
        s->rpm[k] = hover;
        env->actions[k] = 2*hover-1;
    }
    for (int t = 0; t < 8; t++) c_step(env);
    float expected[3] = {.03f, -.02f, .01f};
    bool settled = env->log.settle_n == 2 && env->log.settle_episodes == 1;
    for (int k = 0; k < 3; k++)
        settled &= fabsf(env->log.settle_bias[k] - expected[k]) < 1e-5f &&
                   env->log.settle_within[k] < 1e-8f;
    check("T19 signed settled statistics", settled,
          "per-episode target-frame offset and within-episode jitter from the late window");
    free_env(env);
}

typedef struct {
    int episodes;
    float ret, length, position_error, velocity_error, settle_error, settle_n, d_action, terminated;
    float applied_d_action, saturation, settle_bias[3], settle_across[3], settle_within[3];
} Replay;

static Replay run_replay(int n, int steps) {
    RaptorEnv* env = make_env(n, g_seed);
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
    Replay r = {.episodes = (int)env->log.n,
                .ret = env->log.episode_return / e,
                .length = env->log.episode_length / e,
                .position_error = env->log.position_error / e,
                .velocity_error = env->log.velocity_error / e,
                .settle_error = env->log.settle_error / (env->log.settle_n > 0 ? env->log.settle_n : 1.0f),
                .settle_n = env->log.settle_n,
                .d_action = env->log.d_action / e,
                .terminated = env->log.terminated / e};
    r.applied_d_action = env->log.applied_d_action / e;
    r.saturation = env->log.saturation / e;
    float episodes = env->log.settle_episodes > 0 ? env->log.settle_episodes : 1;
    for (int k = 0; k < 3; k++) {
        r.settle_bias[k] = env->log.settle_bias[k] / episodes;
        r.settle_across[k] = sqrtf(fmaxf(0, env->log.settle_mean_square[k] / episodes
                                            - r.settle_bias[k] * r.settle_bias[k]));
        r.settle_within[k] = sqrtf(env->log.settle_within[k] / episodes);
    }
    free(policies);
    free_env(env);
    return r;
}

static void test_base_policy_replay(void) {
    RaptorEnv saved = g_settings;
    g_settings.dr = 0;
    g_settings.traj.moving_fraction = 0;
    g_settings.yaw_face_probability = 0;
    Replay r = run_replay(256, 5000);
    g_settings = saved;
    printf("     episodes %d | return %.2f | length %.1f | pos err %.3f m | settle %.4f m | "
           "terminated %.1f%%\n",
           r.episodes, r.ret, r.length, r.position_error, r.settle_error, 100.0f * r.terminated);
    char buf[96];
    snprintf(buf, sizeof(buf), "return %.2f, length %.1f, terminated %.1f%%", r.ret, r.length,
             100.0f * r.terminated);
    check("T11 nominal base-policy replay", r.terminated < 0.05f && r.length > 450.0f, buf);
}

static void dump_trajectories(int episodes) {
    RaptorEnv* env = make_env(episodes, g_seed);
    PolicyState* policies = (PolicyState*)calloc(episodes, sizeof(PolicyState));
    State* pre = (State*)calloc(episodes, sizeof(State));
    c_reset(env);
    for (int i = 0; i < episodes; i++) policy_reset(&policies[i]);
    printf("episode,step,x,y,z,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,dist,reward,a0,a1,a2,a3,terminal,tx,ty,tz,truncated\n");
    for (int t = 0; t < env->horizon; t++) {
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
            printf("%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.0f,%.5f,%.5f,%.5f,%.0f\n",
                   i, t, s->position[0], s->position[1], s->position[2], s->linear_velocity[0],
                   s->linear_velocity[1], s->linear_velocity[2], s->angular_velocity[0],
                   s->angular_velocity[1], s->angular_velocity[2], s->orientation[0],
                   s->orientation[1], s->orientation[2], s->orientation[3], d,
                   env->rewards[i], a[0], a[1], a[2], a[3], env->terminals[i] * (1 - env->truncations[i]),
                   s->target[0], s->target[1], s->target[2], env->truncations[i]);
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
    g_settings.reward_params = REWARD_FOUNDATION;
    g_settings.term_params = TERMINATION_FOUNDATION;
    g_settings.init_params = INIT_90_DEG;
    g_settings.horizon = RAPTOR_HORIZON;
    g_settings.obs_clip = 1;
    g_settings.dr = 1;
    g_settings.traj = (TrajParams){0.5f, 20.0f, 0.25f, 15.0f, 0.5f, 2.0f, 0.5f, 0.5f, 0};
    g_settings.yaw_face_probability = 0.5f;
    g_settings.yaw_rate = 0.4f;
    float orientation_deg = 57.29578f;
    struct { const char* name; float* value; } options[] = {
        {"--dr", &g_settings.dr},
        {"--reward-scale", &g_settings.reward_params.scale},
        {"--reward-constant", &g_settings.reward_params.constant},
        {"--reward-position", &g_settings.reward_params.position},
        {"--reward-position-clip", &g_settings.reward_params.position_clip},
        {"--reward-orientation", &g_settings.reward_params.orientation},
        {"--reward-d-action", &g_settings.reward_params.d_action},
        {"--reward-angular-acceleration", &g_settings.reward_params.angular_acceleration},
        {"--term-position", &g_settings.term_params.position},
        {"--term-linear-velocity", &g_settings.term_params.linear_velocity},
        {"--term-angular-velocity", &g_settings.term_params.angular_velocity},
        {"--init-orientation-deg", &orientation_deg},
        {"--init-guidance", &g_settings.init_params.guidance},
        {"--init-position", &g_settings.init_params.max_position},
        {"--init-linear-velocity", &g_settings.init_params.max_linear_velocity},
        {"--init-angular-velocity", &g_settings.init_params.max_angular_velocity},
        {"--init-min-rpm", &g_settings.init_params.min_rpm},
        {"--init-max-rpm", &g_settings.init_params.max_rpm},
        {"--traj-amplitude", &g_settings.traj.amplitude},
        {"--traj-amplitude-min", &g_settings.traj.amplitude_min},
        {"--traj-period", &g_settings.traj.period},
        {"--traj-period-min", &g_settings.traj.period_min},
        {"--traj-moving-fraction", &g_settings.traj.moving_fraction},
        {"--traj-circuit-fraction", &g_settings.traj.circuit_fraction},
        {"--traj-aspect", &g_settings.traj.aspect},
        {"--traj-ramp", &g_settings.traj.ramp},
        {"--yaw-face-probability", &g_settings.yaw_face_probability},
        {"--yaw-rate", &g_settings.yaw_rate},
    };
    const char* mode = NULL;
    const char* mode_args[3] = {NULL, NULL, NULL};
    int n_mode_args = 0;
    for (int i = 1; i < argc; i++) {
        bool setting = false;
        for (size_t k = 0; k < sizeof(options) / sizeof(options[0]); k++) {
            if (strcmp(argv[i], options[k].name) != 0) continue;
            if (i + 1 == argc) { fprintf(stderr, "Missing value for %s\n", argv[i]); return 1; }
            char* end;
            *options[k].value = strtof(argv[++i], &end);
            if (end == argv[i] || *end || !isfinite(*options[k].value)) { fprintf(stderr, "Invalid scalar %s\n", argv[i]); return 1; }
            setting = true;
            break;
        }
        if (setting) continue;
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            policy_load(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            g_seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--airframe") == 0 && i + 1 < argc) {
            g_airframe = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--horizon") == 0 && i + 1 < argc) {
            g_settings.horizon = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--traj-original") == 0 && i + 1 < argc) {
            g_settings.traj.original = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--obs-clip") == 0 && i + 1 < argc) {
            g_settings.obs_clip = atoi(argv[++i]);
        } else if (strncmp(argv[i], "--", 2) == 0) {
            mode = argv[i];
            if (strcmp(mode, "--export-weights") && strcmp(mode, "--fixture-out") &&
                strcmp(mode, "--policy-eval") && strcmp(mode, "--replay") && strcmp(mode, "--dump")) {
                fprintf(stderr, "Unknown option %s\n", mode);
                return 1;
            }
            n_mode_args = 0;
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0 && n_mode_args < 3)
                mode_args[n_mode_args++] = argv[++i];
        } else {
            fprintf(stderr, "Unexpected argument %s\n", argv[i]);
            return 1;
        }
    }
    g_settings.init_params.max_angle = orientation_deg * 3.14159265358979f / 180.0f;
    if (g_airframe < 0 || g_airframe >= (int)(sizeof(AIRFRAMES) / sizeof(AIRFRAMES[0])) ||
        g_settings.horizon <= 0 || (g_settings.obs_clip != 0 && g_settings.obs_clip != 1) ||
        (g_settings.traj.original != 0 && g_settings.traj.original != 1)) {
        fprintf(stderr, "Invalid airframe, horizon, observation clipping or trajectory mode\n");
        return 1;
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
               "\"velocity_error\": %.6f, \"settle_error\": %.6f, \"settle_n\": %.0f, "
               "\"d_action\": %.6f, \"terminated\": %.6f, "
               "\"applied_d_action\": %.6f, \"saturation\": %.6f, "
               "\"settle_bias_x\": %.6f, \"settle_bias_y\": %.6f, \"settle_bias_z\": %.6f, "
               "\"settle_across_x\": %.6f, \"settle_across_y\": %.6f, \"settle_across_z\": %.6f, "
               "\"settle_within_x\": %.6f, \"settle_within_y\": %.6f, \"settle_within_z\": %.6f}\n",
               r.episodes, r.ret, r.length, r.position_error, r.velocity_error, r.settle_error,
               r.settle_n, r.d_action, r.terminated, r.applied_d_action, r.saturation,
               r.settle_bias[0], r.settle_bias[1], r.settle_bias[2],
               r.settle_across[0], r.settle_across[1], r.settle_across[2],
               r.settle_within[0], r.settle_within[1], r.settle_within[2]);
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
    test_native_contract();
    test_command_continuity();
    test_timeout_and_applied_action();
    test_determinism();
    test_policy_parity();
    test_base_policy_replay();
    test_throughput();
    printf("\n%s (%d failures)\n\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures != 0;
}
