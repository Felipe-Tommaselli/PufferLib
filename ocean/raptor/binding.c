#include "raptor.h"

#define OBS_SIZE RAPTOR_OBS_DIM
#define NUM_ATNS RAPTOR_ACT_DIM
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env RaptorEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_agents")->value;
    env->horizon = (int)dict_get(kwargs, "horizon")->value;
    env->obs_clip = (int)dict_get(kwargs, "obs_clip")->value;
    env->dr = dict_get(kwargs, "dr")->value;

    env->airframe = ((int)dict_get(kwargs, "airframe")->value == 1) ? AIRFRAME_CRAZYFLIE : AIRFRAME_X500;

    env->reward_params = REWARD_FOUNDATION;
    env->reward_params.scale = dict_get(kwargs, "reward_scale")->value;
    env->reward_params.constant = dict_get(kwargs, "reward_constant")->value;
    env->reward_params.position = dict_get(kwargs, "reward_position")->value;
    env->reward_params.orientation = dict_get(kwargs, "reward_orientation")->value;
    env->reward_params.d_action = dict_get(kwargs, "reward_d_action")->value;

    env->term_params = TERMINATION_FOUNDATION;
    env->term_params.position = dict_get(kwargs, "term_position")->value;
    env->term_params.linear_velocity = dict_get(kwargs, "term_linear_velocity")->value;
    env->term_params.angular_velocity = dict_get(kwargs, "term_angular_velocity")->value;

    env->init_params = INIT_90_DEG;
    env->init_params.max_angle = dict_get(kwargs, "init_orientation_deg")->value * 3.14159265358979f / 180.0f;
    env->traj.amplitude = dict_get(kwargs, "traj_amplitude")->value;
    env->traj.period = dict_get(kwargs, "traj_period")->value;

    init(env);
}

void my_log(Log* log, Dict* out) {
    float n = log->n > 0.0f ? log->n : 1.0f;
    dict_set(out, "episode_return", log->episode_return / n);
    dict_set(out, "episode_length", log->episode_length / n);
    dict_set(out, "position_error", log->position_error / n);
    dict_set(out, "d_action", log->d_action / n);
    dict_set(out, "terminated", log->terminated / n);
    dict_set(out, "n", log->n);
}
