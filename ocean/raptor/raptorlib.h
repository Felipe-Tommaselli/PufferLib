#pragma once
#include <math.h>
#include <stdbool.h>
#include "airframe.h"

typedef struct {
    float position[3];
    float orientation[4];
    float linear_velocity[3];
    float angular_velocity[3];
    float rpm[4];
    float last_action[4];
    float target[3];
    float target_velocity[3];
} State;

typedef struct {
    float amplitude;
    float period;
} TrajParams;

static inline void trajectory(const TrajParams* tr, float t, float* p, float* v) {
    float w = (tr->amplitude > 0 && tr->period > 0) ? 2.0f * 3.14159265358979f / tr->period : 0.0f;
    p[0] = tr->amplitude * sinf(w * t);
    p[1] = 0.5f * tr->amplitude * sinf(2.0f * w * t);
    p[2] = 0.0f;
    v[0] = tr->amplitude * w * cosf(w * t);
    v[1] = tr->amplitude * w * cosf(2.0f * w * t);
    v[2] = 0.0f;
}

typedef struct {
    float scale;
    float constant;
    float termination_penalty;
    float position;
    float position_clip;
    float orientation;
    float linear_velocity;
    float angular_velocity;
    float linear_acceleration;
    float angular_acceleration;
    float action;
    float d_action;
    bool non_negative;
} RewardParams;

typedef struct {
    bool enabled;
    float position;
    float linear_velocity;
    float angular_velocity;
} TerminationParams;

typedef struct {
    float guidance;
    float max_position;
    float max_angle;
    float max_linear_velocity;
    float max_angular_velocity;
    bool relative_rpm;
    float min_rpm;
    float max_rpm;
} InitParams;

static const RewardParams REWARD_FOUNDATION = {
    1.0f, 1.5f, -100.0f, 1.0f, 0.2f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, false,
};

static const TerminationParams TERMINATION_FOUNDATION = {true, 1.0f, 2.0f, 35.0f};
// upstream sample_orientation ignores its limit argument and draws U(0,1) rad, so init_90_deg is
// really ~57 deg; 1.0 reproduces the distribution the shipped policy was trained on
static const InitParams INIT_90_DEG = {
    0.1f, 0.5f, 1.0f, 1.0f, 1.0f, true, -1.0f, 0.0f,
};

static inline void cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static inline void matvec3(const float m[3][3], const float v[3], float out[3]) {
    for (int i = 0; i < 3; i++) out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
}

static inline void quat_derivative(const float q[4], const float w[3], float out[4]) {
    out[0] = 0.5f * (-q[1] * w[0] - q[2] * w[1] - q[3] * w[2]);
    out[1] = 0.5f * (q[0] * w[0] + q[2] * w[2] - q[3] * w[1]);
    out[2] = 0.5f * (q[0] * w[1] + q[3] * w[0] - q[1] * w[2]);
    out[3] = 0.5f * (q[0] * w[2] + q[1] * w[1] - q[2] * w[0]);
}

static inline void rotate_by_quat(const float q[4], const float v[3], float out[3]) {
    float t[3];
    cross3(&q[1], v, t);
    for (int i = 0; i < 3; i++) t[i] *= 2.0f;
    cross3(&q[1], t, out);
    for (int i = 0; i < 3; i++) out[i] += q[0] * t[i] + v[i];
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float rnd_uniform(unsigned int* rng, float lo, float hi) {
    return lo + (hi - lo) * ((float)rand_r(rng) / (float)RAND_MAX);
}

static inline float rnd_normal(unsigned int* rng) {
    float u1 = rnd_uniform(rng, 1e-7f, 1.0f);
    float u2 = rnd_uniform(rng, 0.0f, 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

static inline float hover_throttle(const Airframe* p) {
    const float* c = p->rotor_thrust_coefficients[0];
    float g = sqrtf(p->gravity[0] * p->gravity[0] + p->gravity[1] * p->gravity[1] +
                    p->gravity[2] * p->gravity[2]);
    float target = p->mass * g / 4.0f - c[0];
    float u = c[2] > 0 ? (-c[1] + sqrtf(c[1] * c[1] + 4 * c[2] * target)) / (2 * c[2])
                       : target / c[1];
    return (u - p->action_limit_min) / (p->action_limit_max - p->action_limit_min);
}

static void randomize_airframe(Airframe* p, float dr, unsigned int* rng) {
    if (dr <= 0) return;
    float hover_nominal = hover_throttle(p);
    p->mass *= rnd_uniform(rng, 1 - 0.30f * dr, 1 + 0.40f * dr);
    for (int k = 0; k < 3; k++) {
        p->J[k][k] *= rnd_uniform(rng, 1 - 0.30f * dr, 1 + 0.40f * dr);
        p->J_inv[k][k] = 1.0f / p->J[k][k];
    }
    float thrust = rnd_uniform(rng, 1 - 0.20f * dr, 1 + 0.20f * dr);
    float torque = rnd_uniform(rng, 1 - 0.20f * dr, 1 + 0.20f * dr);
    float tau_r = rnd_uniform(rng, 1 - 0.40f * dr, 1 + 0.60f * dr);
    float tau_f = rnd_uniform(rng, 1 - 0.40f * dr, 1 + 0.60f * dr);
    float arm = rnd_uniform(rng, 1 - 0.10f * dr, 1 + 0.10f * dr);
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 3; k++) {
            p->rotor_thrust_coefficients[i][k] *= thrust;
            p->rotor_positions[i][k] *= arm;
        }
        p->rotor_torque_constants[i] *= torque;
        p->rotor_time_constants_rising[i] *= tau_r;
        p->rotor_time_constants_falling[i] *= tau_f;
    }
    p->hovering_throttle_relative *= hover_throttle(p) / hover_nominal;
}
