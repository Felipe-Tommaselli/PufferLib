#include "raptor.h"

#define OBS_SIZE RAPTOR_OBS_DIM
#define NUM_ATNS RAPTOR_ACT_DIM
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define MY_TRUNCATION
#define Env RaptorEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_agents")->value;
    env->horizon = (int)dict_get(kwargs, "horizon")->value;
    env->obs_clip = (int)dict_get(kwargs, "obs_clip")->value;
    env->dr = dict_get(kwargs, "dr")->value;

    env->airframe = AIRFRAMES[(int)dict_get(kwargs, "airframe")->value];

    env->reward_params = REWARD_FOUNDATION;
    env->reward_params.scale = dict_get(kwargs, "reward_scale")->value;
    env->reward_params.constant = dict_get(kwargs, "reward_constant")->value;
    env->reward_params.position = dict_get(kwargs, "reward_position")->value;
    env->reward_params.position_clip = dict_get(kwargs, "reward_position_clip")->value;
    env->reward_params.orientation = dict_get(kwargs, "reward_orientation")->value;
    env->reward_params.d_action = dict_get(kwargs, "reward_d_action")->value;
    env->reward_params.angular_acceleration = dict_get(kwargs, "reward_angular_acceleration")->value;

    env->term_params = TERMINATION_FOUNDATION;
    env->term_params.position = dict_get(kwargs, "term_position")->value;
    env->term_params.linear_velocity = dict_get(kwargs, "term_linear_velocity")->value;
    env->term_params.angular_velocity = dict_get(kwargs, "term_angular_velocity")->value;

    env->init_params = INIT_90_DEG;
    env->init_params.max_angle = dict_get(kwargs, "init_orientation_deg")->value * 3.14159265358979f / 180.0f;
    env->init_params.guidance = dict_get(kwargs, "init_guidance")->value;
    env->init_params.max_position = dict_get(kwargs, "init_position")->value;
    env->init_params.max_linear_velocity = dict_get(kwargs, "init_linear_velocity")->value;
    env->init_params.max_angular_velocity = dict_get(kwargs, "init_angular_velocity")->value;
    env->init_params.min_rpm = dict_get(kwargs, "init_min_rpm")->value;
    env->init_params.max_rpm = dict_get(kwargs, "init_max_rpm")->value;
    env->traj.original = (int)dict_get(kwargs, "traj_original")->value;
    env->traj.moving_fraction = dict_get(kwargs, "traj_moving_fraction")->value;

    init(env);
}

void my_log(Log* log, Dict* out) {
    float n = log->n > 0.0f ? log->n : 1.0f;
    dict_set(out, "episode_return", log->episode_return / n);
    dict_set(out, "episode_length", log->episode_length / n);
    dict_set(out, "position_error", log->position_error / n);
    dict_set(out, "velocity_error", log->velocity_error / n);
    dict_set(out, "settle_error", log->settle_error / (log->settle_n > 0.0f ? log->settle_n : 1.0f));
    dict_set(out, "settle_n", log->settle_n / n);
    dict_set(out, "d_action", log->d_action / n);
    dict_set(out, "applied_d_action", log->applied_d_action / n);
    dict_set(out, "saturation", log->saturation / n);
    const char* bias_keys[] = {"settle_bias_x", "settle_bias_y", "settle_bias_z"};
    const char* across_keys[] = {"settle_across_x", "settle_across_y", "settle_across_z"};
    const char* within_keys[] = {"settle_within_x", "settle_within_y", "settle_within_z"};
    float episodes = log->settle_episodes > 0 ? log->settle_episodes : 1;
    for (int k = 0; k < 3; k++) {
        float bias = log->settle_bias[k] / episodes;
        dict_set(out, bias_keys[k], bias);
        dict_set(out, across_keys[k],
                 sqrtf(fmaxf(0, log->settle_mean_square[k] / episodes - bias * bias)));
        dict_set(out, within_keys[k], sqrtf(log->settle_within[k] / episodes));
    }
    dict_set(out, "terminated", log->terminated / n);
    dict_set(out, "n", log->n);
}
