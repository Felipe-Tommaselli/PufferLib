#pragma once
#include <stdlib.h>
#include <string.h>
#include "task_hover.h"
#include "physics.h"

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float position_error;
    float d_action;
    float terminated;
    float n;
};

typedef struct {
    State state;
    Airframe airframe;
    float episode_return;
    int episode_length;
    float position_error_sum;
    float d_action_sum;
} Agent;

typedef struct RaptorEnv RaptorEnv;
struct RaptorEnv {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
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
};

static void add_log(RaptorEnv* env, Agent* agent, bool term) {
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.position_error += agent->position_error_sum / agent->episode_length;
    env->log.d_action += agent->d_action_sum / agent->episode_length;
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
    reset_state(&agent->airframe, &env->init_params, &agent->state, &env->rng);
    trajectory(&env->traj, 0.0f, agent->state.target, agent->state.target_velocity);
}

void c_reset(RaptorEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        reset_agent(env, i);
        observe(&env->agents[i].state, env->observations + i * RAPTOR_OBS_DIM, env->obs_clip);
    }
}

void c_step(RaptorEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        Agent* agent = &env->agents[i];
        float* action = env->actions + i * RAPTOR_ACT_DIM;

        State next;
        physics_step(&agent->airframe, &agent->state, action, &next);
        trajectory(&env->traj, (agent->episode_length + 1) * RAPTOR_DT, next.target, next.target_velocity);
        bool term = terminated(&env->term_params, &next);
        float r = reward(&agent->airframe, &env->reward_params, &agent->state, action, &next, term);

        float d_action = 0;
        for (int k = 0; k < RAPTOR_ACT_DIM; k++) {
            float d = action[k] - agent->state.last_action[k];
            d_action += d * d;
        }
        agent->d_action_sum += sqrtf(d_action);
        float ex = next.position[0] - next.target[0], ey = next.position[1] - next.target[1],
              ez = next.position[2] - next.target[2];
        agent->position_error_sum += sqrtf(ex * ex + ey * ey + ez * ez);

        agent->state = next;
        agent->episode_return += r;
        agent->episode_length += 1;

        bool truncated = agent->episode_length >= env->horizon;
        env->rewards[i] = r;
        env->terminals[i] = (term || truncated) ? 1.0f : 0.0f;

        if (term || truncated) {
            add_log(env, agent, term);
            reset_agent(env, i);
        }
        observe(&agent->state, env->observations + i * RAPTOR_OBS_DIM, env->obs_clip);
    }
}

void init(RaptorEnv* env) {
    env->agents = (Agent*)calloc(env->num_agents, sizeof(Agent));
    if (env->horizon == 0) env->horizon = RAPTOR_HORIZON;
    memset(&env->log, 0, sizeof(Log));
}

void c_render(RaptorEnv* env) { (void)env; }

void c_close(RaptorEnv* env) { free(env->agents); }
