#pragma once
#include "raptorlib.h"

static void observe(const State* s, float* obs, bool clip) {
    const float* q = s->orientation;
    float pos[3], vel[3];
    for (int i = 0; i < 3; i++) {
        pos[i] = s->position[i] - s->target[i];
        vel[i] = s->linear_velocity[i] - s->target_velocity[i];
    }
    float c = cosf(s->target_yaw), sn = sinf(s->target_yaw);
    float x = pos[0], vx = vel[0];
    pos[0] = c * x + sn * pos[1];
    pos[1] = -sn * x + c * pos[1];
    vel[0] = c * vx + sn * vel[1];
    vel[1] = -sn * vx + c * vel[1];
    for (int i = 0; i < 3 && clip; i++) {
        pos[i] = clampf(pos[i], -0.1f, 0.1f);
        vel[i] = clampf(vel[i], -1.0f, 1.0f);
    }
    obs[0] = pos[0];
    obs[1] = pos[1];
    obs[2] = pos[2];
    obs[3] = 1 - 2 * q[2] * q[2] - 2 * q[3] * q[3];
    obs[4] = 2 * q[1] * q[2] - 2 * q[0] * q[3];
    obs[5] = 2 * q[1] * q[3] + 2 * q[0] * q[2];
    obs[6] = 2 * q[1] * q[2] + 2 * q[0] * q[3];
    obs[7] = 1 - 2 * q[1] * q[1] - 2 * q[3] * q[3];
    obs[8] = 2 * q[2] * q[3] - 2 * q[0] * q[1];
    obs[9] = 2 * q[1] * q[3] - 2 * q[0] * q[2];
    obs[10] = 2 * q[2] * q[3] + 2 * q[0] * q[1];
    obs[11] = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2];
    for (int i = 0; i < 3; i++) {
        float row0 = obs[3 + i], row1 = obs[6 + i];
        obs[3 + i] = c * row0 + sn * row1;
        obs[6 + i] = -sn * row0 + c * row1;
    }
    obs[12] = vel[0];
    obs[13] = vel[1];
    obs[14] = vel[2];
    obs[15] = s->angular_velocity[0];
    obs[16] = s->angular_velocity[1];
    obs[17] = s->angular_velocity[2];
    for (int i = 0; i < 4; i++) obs[18 + i] = s->last_action[i];
}

static bool terminated(const TerminationParams* t, const State* s) {
    for (int i = 0; i < 3; i++)
        if (isnan(s->position[i]) || isnan(s->linear_velocity[i]) || isnan(s->angular_velocity[i])) return true;
    if (!t->enabled) return false;
    for (int i = 0; i < 3; i++) {
        if (fabsf(s->position[i] - s->target[i]) > t->position) return true;
        if (fabsf(s->linear_velocity[i] - s->target_velocity[i]) > t->linear_velocity) return true;
        if (fabsf(s->angular_velocity[i]) > t->angular_velocity) return true;
    }
    return false;
}

static float reward(const Airframe* p, const RewardParams* r, const State* s, const float* action,
                    const State* next, bool term) {
    if (term) return r->termination_penalty;

    float px = s->position[0] - s->target[0], py = s->position[1] - s->target[1],
          pz = s->position[2] - s->target[2];
    float position_cost = sqrtf(px * px + py * py + pz * pz);
    if (r->position_clip > 0 && position_cost > r->position_clip) position_cost = r->position_clip;

    float relative_z = cosf(0.5f * s->target_yaw) * s->orientation[3] -
                       sinf(0.5f * s->target_yaw) * s->orientation[0];
    float orientation_cost = 2 * acosf(clampf(1 - fabsf(relative_z), -1.0f, 1.0f));

    float vx = s->linear_velocity[0] - s->target_velocity[0],
          vy = s->linear_velocity[1] - s->target_velocity[1],
          vz = s->linear_velocity[2] - s->target_velocity[2];
    float linear_vel_cost = sqrtf(vx * vx + vy * vy + vz * vz);
    float angular_vel_cost = sqrtf(s->angular_velocity[0] * s->angular_velocity[0] +
                                   s->angular_velocity[1] * s->angular_velocity[1] +
                                   s->angular_velocity[2] * s->angular_velocity[2]);

    float linear_acc = 0, angular_acc = 0;
    for (int i = 0; i < 3; i++) {
        float dv = next->linear_velocity[i] - s->linear_velocity[i];
        float dw = next->angular_velocity[i] - s->angular_velocity[i];
        linear_acc += dv * dv;
        angular_acc += dw * dw;
    }
    float linear_acc_cost = sqrtf(linear_acc) / RAPTOR_DT;
    float angular_acc_cost = sqrtf(angular_acc) / RAPTOR_DT;

    float action_cost = 0, d_action_cost = 0;
    for (int i = 0; i < 4; i++) {
        float rel = (action[i] + 1.0f) / 2.0f - p->hovering_throttle_relative;
        action_cost += rel * rel;
        float d = action[i] - s->last_action[i];
        d_action_cost += d * d;
    }
    d_action_cost = sqrtf(d_action_cost);

    float cost = r->position * position_cost + r->orientation * orientation_cost +
                 r->linear_velocity * linear_vel_cost + r->angular_velocity * angular_vel_cost +
                 r->linear_acceleration * linear_acc_cost + r->angular_acceleration * angular_acc_cost +
                 r->action * action_cost + r->d_action * d_action_cost;

    float value = r->constant - r->scale * cost;
    return (value > 0 || !r->non_negative) ? value : 0;
}

static void reset_state(const Airframe* p, const InitParams* in, State* s, unsigned int* rng) {
    bool guidance = rnd_uniform(rng, 0.0f, 1.0f) < in->guidance;
    for (int i = 0; i < 3; i++) {
        s->position[i] = guidance ? 0.0f : rnd_uniform(rng, -in->max_position, in->max_position);
        s->linear_velocity[i] = guidance ? 0.0f : rnd_uniform(rng, -in->max_linear_velocity, in->max_linear_velocity);
        s->angular_velocity[i] = guidance ? 0.0f : rnd_uniform(rng, -in->max_angular_velocity, in->max_angular_velocity);
    }
    for (int i = 0; i < 3; i++) s->target[i] = s->target_velocity[i] = 0.0f;
    s->target_yaw = 0;
    if (in->max_angle > 0 && !guidance) {
        float u = rnd_uniform(rng, 0.0f, 1.0f);
        float v = rnd_uniform(rng, 0.0f, 1.0f);
        float phi = 2.0f * 3.14159265358979f * u;
        float cos_theta = 1.0f - 2.0f * v;
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
        float half = 0.5f * rnd_uniform(rng, 0.0f, in->max_angle);
        float sn = sinf(half);
        s->orientation[0] = cosf(half);
        s->orientation[1] = sin_theta * cosf(phi) * sn;
        s->orientation[2] = sin_theta * sinf(phi) * sn;
        s->orientation[3] = cos_theta * sn;
    } else {
        s->orientation[0] = 1;
        s->orientation[1] = 0;
        s->orientation[2] = 0;
        s->orientation[3] = 0;
    }
    float lo = in->min_rpm, hi = in->max_rpm;
    if (in->relative_rpm) {
        float range = p->action_limit_max - p->action_limit_min;
        lo = (in->min_rpm + 1) / 2 * range + p->action_limit_min;
        hi = (in->max_rpm + 1) / 2 * range + p->action_limit_min;
    }
    for (int i = 0; i < 4; i++) {
        s->rpm[i] = rnd_uniform(rng, lo, hi);
        s->last_action[i] = 0.0f;
    }
}
