#pragma once

#define RAPTOR_DT 0.01f
#define RAPTOR_HORIZON 500
#define RAPTOR_OBS_DIM 22
#define RAPTOR_ACT_DIM 4

typedef struct {
    float rotor_positions[4][3];
    float rotor_thrust_directions[4][3];
    float rotor_torque_directions[4][3];
    float rotor_thrust_coefficients[4][3];
    float rotor_torque_constants[4];
    float rotor_time_constants_rising[4];
    float rotor_time_constants_falling[4];
    float mass;
    float gravity[3];
    float J[3][3];
    float J_inv[3][3];
    float hovering_throttle_relative;
    float action_limit_min;
    float action_limit_max;
} Airframe;

// rotors in Crazyflie order; x500_real.h is PX4 order, permuted 0,3,1,2 as the registry does
static const Airframe AIRFRAME_X500 = {
    {{+0.176776695296636f, -0.176776695296636f, 0.0f},
     {-0.176776695296636f, -0.176776695296636f, 0.0f},
     {-0.176776695296636f, +0.176776695296636f, 0.0f},
     {+0.176776695296636f, +0.176776695296636f, 0.0f}},
    {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}},
    {{0, 0, -1}, {0, 0, +1}, {0, 0, -1}, {0, 0, +1}},
    {{0, 0, 15.926119900619797f},
     {0, 0, 15.926119900619797f},
     {0, 0, 15.926119900619797f},
     {0, 0, 15.926119900619797f}},
    {1.0847e-1f, 1.0847e-1f, 1.0847e-1f, 1.0847e-1f},
    {0.054f, 0.054f, 0.054f, 0.054f},
    {0.054f, 0.054f, 0.054f, 0.054f},
    2.000f,
    {0, 0, -9.81f},
    {{2.1705e-2f, 0.0f, 0.0f}, {0.0f, 2.1304e-2f, 0.0f}, {0.0f, 0.0f, 0.039396244f}},
    {{46.072333563695004f, 0.0f, 0.0f}, {0.0f, 46.939541870071345f, 0.0f}, {0.0f, 0.0f, 25.38313043243412f}},
    0.5549636212401485f,
    0.0f,
    1.0f,
};

static const Airframe AIRFRAME_CRAZYFLIE = {
    {{+0.028f, -0.028f, 0.0f}, {-0.028f, -0.028f, 0.0f}, {-0.028f, +0.028f, 0.0f}, {+0.028f, +0.028f, 0.0f}},
    {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}},
    {{0, 0, -1}, {0, 0, +1}, {0, 0, -1}, {0, 0, +1}},
    {{0.00352526f, 0.01437313f, 0.092230483f},
     {0.00352526f, 0.01437313f, 0.092230483f},
     {0.00352526f, 0.01437313f, 0.092230483f},
     {0.00352526f, 0.01437313f, 0.092230483f}},
    {4.665e-3f, 4.665e-3f, 4.665e-3f, 4.665e-3f},
    {0.0555f, 0.0555f, 0.0555f, 0.0555f},
    {0.2494f, 0.2494f, 0.2494f, 0.2494f},
    0.027f,
    {0, 0, -9.81f},
    {{9.4166e-06f, 0.0f, 0.0f}, {0.0f, 9.4166e-06f, 0.0f}, {0.0f, 0.0f, 1.4166e-05f}},
    {{106195.24f, 0.0f, 0.0f}, {0.0f, 106195.24f, 0.0f}, {0.0f, 0.0f, 70591.55f}},
    0.7261389721508553f,
    0.0f,
    1.0f,
};
