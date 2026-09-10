#pragma once
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
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
    float target_yaw;
} State;

typedef struct {
    float amplitude;
    float period;
    float amplitude_min;
    float period_min;
    float moving_fraction;
    float ramp;
    float circuit_fraction;
    float aspect;
    int original;
} TrajParams;

typedef struct {
    float amplitude, period, phase, direction, heading, ramp;
    bool face_travel;
    bool circuit;
    bool moving;
    bool original;
    float aspect;
    float position[3];
    float velocity[3];
    float raw_position[3];
    float raw_velocity[3];
} Trajectory;

static inline float rnd_normal(unsigned int* rng);

static inline void trajectory(const Trajectory* tr, float t, float* p, float* v) {
    float clock = t, speed = 1.0f;
    if (tr->ramp > 0) {
        float u = fminf(t / tr->ramp, 1.0f);
        speed = u * u * (3.0f - 2.0f * u);
        clock = t < tr->ramp ? tr->ramp * u * u * u * (1.0f - 0.5f * u) : t - 0.5f * tr->ramp;
    }
    float w = tr->amplitude > 0 ? tr->direction * 2.0f * 3.14159265358979f / tr->period : 0;
    float phase = tr->phase + w * clock;
    float x, y, vx, vy;
    if (tr->circuit) {
        float segment = phase * (2.0f / 3.14159265358979f);
        float whole = floorf(segment), u = segment - whole;
        int start = ((int)fmodf(whole, 4.0f) + 4) % 4;
        float basis[4] = {(1-u)*(1-u)*(1-u)/6, (3*u*u*u-6*u*u+4)/6,
                          (-3*u*u*u+3*u*u+3*u+1)/6, u*u*u/6};
        float derivative[4] = {-0.5f*(1-u)*(1-u), 1.5f*u*u-2*u,
                               -1.5f*u*u+u+0.5f, 0.5f*u*u};
        static const float corners[4][2] = {{-1,-1}, {1,-1}, {1,1}, {-1,1}};
        x = y = vx = vy = 0;
        for (int i = 0; i < 4; i++) {
            const float* point = corners[(start + i) % 4];
            x += basis[i] * point[0];
            y += basis[i] * point[1];
            vx += derivative[i] * point[0];
            vy += derivative[i] * point[1];
        }
        x *= tr->amplitude;
        y *= tr->amplitude * tr->aspect;
        vx *= tr->amplitude * w * speed * (2.0f / 3.14159265358979f);
        vy *= tr->amplitude * tr->aspect * w * speed * (2.0f / 3.14159265358979f);
    } else {
        x = tr->amplitude * sinf(phase);
        y = tr->aspect * tr->amplitude * sinf(2.0f * phase);
        vx = tr->amplitude * w * speed * cosf(phase);
        vy = 2.0f * tr->aspect * tr->amplitude * w * speed * cosf(2.0f * phase);
    }
    float c = cosf(tr->heading), s = sinf(tr->heading);
    p[0] = c * x - s * y;
    p[1] = s * x + c * y;
    p[2] = 0;
    v[0] = c * vx - s * vy;
    v[1] = s * vx + c * vy;
    v[2] = 0;
}

static inline void original_trajectory(Trajectory* tr, float* p, float* v, unsigned int* rng) {
    if (tr->moving) {
        for (int i = 0; i < 3; i++) {
            tr->raw_velocity[i] += (-tr->raw_velocity[i] - 4.0f * tr->raw_position[i]) * RAPTOR_DT
                                 + 0.5f * sqrtf(RAPTOR_DT) * rnd_normal(rng);
            tr->raw_position[i] += tr->raw_velocity[i] * RAPTOR_DT;
            tr->velocity[i] = 0.01f * tr->raw_velocity[i] + 0.99f * tr->velocity[i];
            tr->position[i] += tr->velocity[i] * RAPTOR_DT;
        }
    }
    memcpy(p, tr->position, sizeof(tr->position));
    memcpy(v, tr->velocity, sizeof(tr->velocity));
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
    1.0f, 1.5f, -100.0f, 1.0f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, false,
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

static inline bool rnd_bernoulli(unsigned int* rng, float probability) {
    return (double)rand_r(rng) < probability * ((double)RAND_MAX + 1.0);
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
    p->mass *= rnd_uniform(rng, 1 - 0.15f * dr, 1 + 0.15f * dr);
    for (int k = 0; k < 3; k++) {
        p->J[k][k] *= rnd_uniform(rng, 1 - 0.04f * dr, 1 + 0.04f * dr);
        p->J_inv[k][k] = 1.0f / p->J[k][k];
    }
    float thrust = rnd_uniform(rng, 1 - 0.15f * dr, 1 + 0.15f * dr);
    float torque = rnd_uniform(rng, 1 - 0.04f * dr, 1 + 0.04f * dr);
    float tau_r = rnd_uniform(rng, 1 - 0.04f * dr, 1 + 0.04f * dr);
    float tau_f = rnd_uniform(rng, 1 - 0.04f * dr, 1 + 0.04f * dr);
    float arm = rnd_uniform(rng, 1 - 0.04f * dr, 1 + 0.04f * dr);
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
