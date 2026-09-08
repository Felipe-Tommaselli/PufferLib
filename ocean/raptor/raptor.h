#pragma once
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "task_hover.h"
#include "physics.h"

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float position_error;
    float settle_error;
    float settle_n;
    float d_action;
    float applied_d_action;
    float saturation;
    float settle_bias[3];
    float settle_mean_square[3];
    float settle_within[3];
    float settle_episodes;
    float terminated;
    float n;
};

typedef struct {
    State state;
    Airframe airframe;
    Trajectory trajectory;
    float episode_return;
    int episode_length;
    float position_error_sum;
    float d_action_sum;
    float applied_d_action_sum;
    float saturation_sum;
    float settle_bias[3];
    float settle_square[3];
    float settle_sum;
    int settle_n;
} Agent;

typedef struct RaptorEnv RaptorEnv;
struct RaptorEnv {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    float* truncations;
    float* final_observations;
    int num_agents;
    unsigned int rng;

    Agent* agents;
    Log log;

    RewardParams reward_params;
    TerminationParams term_params;
    InitParams init_params;
    Airframe airframe;
    int horizon;
    int obs_clip;
    float dr;
    TrajParams traj;
    float yaw_face_probability;
    float yaw_rate;
};

static void add_log(RaptorEnv* env, Agent* agent, bool term) {
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.position_error += agent->position_error_sum / agent->episode_length;
    env->log.settle_error += agent->settle_sum;
    env->log.settle_n += agent->settle_n;
    env->log.d_action += agent->d_action_sum / agent->episode_length;
    env->log.applied_d_action += agent->applied_d_action_sum / agent->episode_length;
    env->log.saturation += agent->saturation_sum / agent->episode_length;
    if (agent->settle_n > 0) {
        for (int k = 0; k < 3; k++) {
            float mean = agent->settle_bias[k] / agent->settle_n;
            env->log.settle_bias[k] += mean;
            env->log.settle_mean_square[k] += mean * mean;
            env->log.settle_within[k] +=
                fmaxf(0, agent->settle_square[k] / agent->settle_n - mean * mean);
        }
        env->log.settle_episodes += 1.0f;
    }
    env->log.terminated += term ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

static void reset_agent(RaptorEnv* env, int idx) {
    Agent* agent = &env->agents[idx];
    agent->airframe = env->airframe;
    randomize_airframe(&agent->airframe, env->dr, &env->rng);
    agent->episode_return = 0;
    agent->episode_length = 0;
    agent->position_error_sum = 0;
    agent->d_action_sum = 0;
    agent->applied_d_action_sum = 0;
    agent->saturation_sum = 0;
    agent->settle_sum = 0;
    agent->settle_n = 0;
    memset(agent->settle_bias, 0, sizeof(agent->settle_bias));
    memset(agent->settle_square, 0, sizeof(agent->settle_square));
    Trajectory* tr = &agent->trajectory;
    tr->amplitude = rnd_bernoulli(&env->rng, env->traj.moving_fraction) ?
        rnd_uniform(&env->rng, env->traj.amplitude_min, env->traj.amplitude) : 0;
    tr->period = rnd_uniform(&env->rng, env->traj.period_min, env->traj.period);
    tr->phase = rnd_uniform(&env->rng, 0, 2.0f * 3.14159265358979f);
    tr->direction = rnd_bernoulli(&env->rng, 0.5f) ? 1.0f : -1.0f;
    tr->heading = rnd_uniform(&env->rng, -3.14159265358979f, 3.14159265358979f);
    tr->ramp = env->traj.ramp;
    tr->aspect = env->traj.aspect;
    tr->face_travel = rnd_bernoulli(&env->rng, env->yaw_face_probability);
    tr->circuit = rnd_bernoulli(&env->rng, env->traj.circuit_fraction);
    reset_state(&agent->airframe, &env->init_params, &agent->state, &env->rng);
    State* s = &agent->state;
    if (tr->face_travel) s->target_yaw = tr->heading;
    trajectory(tr, 0.0f, s->target, s->target_velocity);
    if (tr->face_travel && tr->amplitude > 0) {
        Trajectory tangent = *tr;
        tangent.ramp = 0;
        float position[3], velocity[3];
        trajectory(&tangent, 0, position, velocity);
        s->target_yaw = atan2f(velocity[1], velocity[0]);
    }
    float c = cosf(s->target_yaw), sn = sinf(s->target_yaw);
    float x = s->position[0], vx = s->linear_velocity[0];
    s->position[0] = c * x - sn * s->position[1];
    s->position[1] = sn * x + c * s->position[1];
    s->linear_velocity[0] = c * vx - sn * s->linear_velocity[1];
    s->linear_velocity[1] = sn * vx + c * s->linear_velocity[1];
    for (int k = 0; k < 3; k++) {
        s->position[k] += s->target[k];
        s->linear_velocity[k] += s->target_velocity[k];
    }
    float ch = cosf(0.5f * s->target_yaw), sh = sinf(0.5f * s->target_yaw);
    float q[4];
    memcpy(q, s->orientation, sizeof(q));
    s->orientation[0] = ch * q[0] - sh * q[3];
    s->orientation[1] = ch * q[1] - sh * q[2];
    s->orientation[2] = ch * q[2] + sh * q[1];
    s->orientation[3] = ch * q[3] + sh * q[0];
}

void c_reset(RaptorEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        reset_agent(env, i);
        env->terminals[i] = 0;
        if (env->truncations) env->truncations[i] = 0;
        observe(&env->agents[i].state, env->observations + i * RAPTOR_OBS_DIM, env->obs_clip);
    }
}

void c_step(RaptorEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        Agent* agent = &env->agents[i];
        float* action = env->actions + i * RAPTOR_ACT_DIM;

        State next;
        physics_step(&agent->airframe, &agent->state, action, &next);
        trajectory(&agent->trajectory, (agent->episode_length + 1) * RAPTOR_DT, next.target, next.target_velocity);
        if (agent->trajectory.face_travel &&
            next.target_velocity[0] * next.target_velocity[0] + next.target_velocity[1] * next.target_velocity[1] > 1e-12f) {
            float desired = atan2f(next.target_velocity[1], next.target_velocity[0]);
            float delta = remainderf(desired - next.target_yaw, 2.0f * 3.14159265358979f);
            next.target_yaw += clampf(delta, -env->yaw_rate * RAPTOR_DT, env->yaw_rate * RAPTOR_DT);
        }
        bool term = terminated(&env->term_params, &next);
        float r = reward(&agent->airframe, &env->reward_params, &agent->state, action, &next, term);

        float d_action = 0, applied_d_action = 0, saturation = 0;
        for (int k = 0; k < RAPTOR_ACT_DIM; k++) {
            float d = action[k] - agent->state.last_action[k];
            d_action += d * d;
            float applied = clampf(action[k], -1.0f, 1.0f);
            float delta = applied - clampf(agent->state.last_action[k], -1.0f, 1.0f);
            applied_d_action += delta * delta;
            saturation += fabsf(action[k]) >= 1.0f ? 0.25f : 0;
        }
        agent->d_action_sum += sqrtf(d_action);
        agent->applied_d_action_sum += sqrtf(applied_d_action);
        agent->saturation_sum += saturation;
        float ex = next.position[0] - next.target[0], ey = next.position[1] - next.target[1],
              ez = next.position[2] - next.target[2];
        float e = sqrtf(ex * ex + ey * ey + ez * ez);
        agent->position_error_sum += e;
        if (agent->episode_length * 4 >= env->horizon * 3) {
            agent->settle_sum += e;
            agent->settle_n += 1;
            float c = cosf(next.target_yaw), sn = sinf(next.target_yaw);
            float error[3] = {c * ex + sn * ey, -sn * ex + c * ey, ez};
            for (int k = 0; k < 3; k++) {
                agent->settle_bias[k] += error[k];
                agent->settle_square[k] += error[k] * error[k];
            }
        }

        agent->state = next;
        agent->episode_return += r;
        agent->episode_length += 1;

        bool truncated = agent->episode_length >= env->horizon;
        env->rewards[i] = r;
        env->terminals[i] = (term || truncated) ? 1.0f : 0.0f;
        if (env->truncations) env->truncations[i] = truncated && !term;
        if (truncated && !term && env->final_observations)
            observe(&agent->state, env->final_observations + i * RAPTOR_OBS_DIM, env->obs_clip);

        if (term || truncated) {
            add_log(env, agent, term);
            reset_agent(env, i);
        }
        observe(&agent->state, env->observations + i * RAPTOR_OBS_DIM, env->obs_clip);
    }
}

void init(RaptorEnv* env) {
    if (!(isfinite(env->dr) && env->dr >= 0 && env->dr < 1.0f / 0.15f &&
          env->traj.amplitude_min >= 0 && env->traj.amplitude >= env->traj.amplitude_min &&
          env->traj.period_min > 0 && env->traj.period >= env->traj.period_min &&
          env->traj.moving_fraction >= 0 && env->traj.moving_fraction <= 1 &&
          env->traj.circuit_fraction >= 0 && env->traj.circuit_fraction <= 1 &&
          env->traj.aspect > 0 &&
          env->traj.ramp >= 0 && env->yaw_face_probability >= 0 && env->yaw_face_probability <= 1 &&
          env->yaw_rate >= 0)) {
        fprintf(stderr, "Invalid DR, trajectory or yaw settings\n");
        abort();
    }
    env->agents = (Agent*)calloc(env->num_agents, sizeof(Agent));
    if (env->horizon == 0) env->horizon = RAPTOR_HORIZON;
    memset(&env->log, 0, sizeof(Log));
}

void c_render(RaptorEnv* env) { (void)env; }

void c_close(RaptorEnv* env) { free(env->agents); }
