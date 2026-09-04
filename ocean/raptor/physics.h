#pragma once
#include "raptorlib.h"

static void dynamics(const Airframe* p, const State* s, const float* action, State* d) {
    float thrust[3] = {0, 0, 0};
    float torque[3] = {0, 0, 0};
    for (int i = 0; i < 4; i++) {
        float u = s->rpm[i];
        const float* c = p->rotor_thrust_coefficients[i];
        float mag = c[0] + c[1] * u + c[2] * u * u;
        float rotor_thrust[3];
        for (int k = 0; k < 3; k++) {
            rotor_thrust[k] = p->rotor_thrust_directions[i][k] * mag;
            thrust[k] += rotor_thrust[k];
            torque[k] += p->rotor_torque_directions[i][k] * mag * p->rotor_torque_constants[i];
        }
        float arm_torque[3];
        cross3(p->rotor_positions[i], rotor_thrust, arm_torque);
        for (int k = 0; k < 3; k++) torque[k] += arm_torque[k];
    }

    for (int k = 0; k < 3; k++) d->position[k] = s->linear_velocity[k];
    quat_derivative(s->orientation, s->angular_velocity, d->orientation);

    rotate_by_quat(s->orientation, thrust, d->linear_velocity);
    for (int k = 0; k < 3; k++) d->linear_velocity[k] = d->linear_velocity[k] / p->mass + p->gravity[k];

    float jw[3], wxjw[3], net[3];
    matvec3(p->J, s->angular_velocity, jw);
    cross3(s->angular_velocity, jw, wxjw);
    for (int k = 0; k < 3; k++) net[k] = torque[k] - wxjw[k];
    matvec3(p->J_inv, net, d->angular_velocity);

    for (int i = 0; i < 4; i++) {
        float tau = action[i] >= s->rpm[i] ? p->rotor_time_constants_rising[i] : p->rotor_time_constants_falling[i];
        d->rpm[i] = (action[i] - s->rpm[i]) / tau;
    }
}

static inline void state_axpy(const State* d, float a, State* acc) {
    for (int k = 0; k < 3; k++) acc->position[k] += a * d->position[k];
    for (int k = 0; k < 4; k++) acc->orientation[k] += a * d->orientation[k];
    for (int k = 0; k < 3; k++) acc->linear_velocity[k] += a * d->linear_velocity[k];
    for (int k = 0; k < 3; k++) acc->angular_velocity[k] += a * d->angular_velocity[k];
    for (int k = 0; k < 4; k++) acc->rpm[k] += a * d->rpm[k];
}

static void rk4(const Airframe* p, const State* s, const float* action, float dt, State* next) {
    State k1, k2, k3, k4, tmp;
    dynamics(p, s, action, &k1);
    tmp = *s;
    state_axpy(&k1, dt / 2, &tmp);
    dynamics(p, &tmp, action, &k2);
    tmp = *s;
    state_axpy(&k2, dt / 2, &tmp);
    dynamics(p, &tmp, action, &k3);
    tmp = *s;
    state_axpy(&k3, dt, &tmp);
    dynamics(p, &tmp, action, &k4);
    *next = *s;
    state_axpy(&k1, dt / 6, next);
    state_axpy(&k2, dt / 3, next);
    state_axpy(&k3, dt / 3, next);
    state_axpy(&k4, dt / 6, next);
}

static void physics_step(const Airframe* p, const State* s, const float* action, State* next) {
    float scaled[4];
    float half = (p->action_limit_max - p->action_limit_min) / 2;
    for (int i = 0; i < 4; i++)
        scaled[i] = clampf(action[i], -1.0f, 1.0f) * half + p->action_limit_min + half;

    rk4(p, s, scaled, RAPTOR_DT, next);

    float norm = 0;
    for (int i = 0; i < 4; i++) norm += next->orientation[i] * next->orientation[i];
    norm = sqrtf(norm);
    for (int i = 0; i < 4; i++) next->orientation[i] /= norm;

    for (int i = 0; i < 4; i++)
        next->rpm[i] = clampf(next->rpm[i], p->action_limit_min, p->action_limit_max);
    for (int i = 0; i < 4; i++) next->last_action[i] = action[i];
}
