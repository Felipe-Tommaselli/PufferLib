#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "policy_weights.h"

#define POLICY_HIDDEN 16
#define POLICY_NPARAM 2084

// canonical flat layout, shared by the blob, the trainer and the harness
static float PW[POLICY_NPARAM];
#define P_W0 (PW + 0)
#define P_B0 (PW + 352)
#define P_WI (PW + 368)
#define P_WH (PW + 1136)
#define P_BI (PW + 1904)
#define P_BH (PW + 1952)
#define P_H0 (PW + 2000)
#define P_W2 (PW + 2016)
#define P_B2 (PW + 2080)

typedef struct {
    float h[POLICY_HIDDEN];
} PolicyState;

static void policy_init(void) {
    memcpy(P_W0, W0, sizeof(W0));
    memcpy(P_B0, B0, sizeof(B0));
    memcpy(P_WI, WI, sizeof(WI));
    memcpy(P_WH, WH, sizeof(WH));
    memcpy(P_BI, BI, sizeof(BI));
    memcpy(P_BH, BH, sizeof(BH));
    memcpy(P_H0, H0, sizeof(H0));
    memcpy(P_W2, W2, sizeof(W2));
    memcpy(P_B2, B2, sizeof(B2));
}

static void policy_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "policy_load: cannot open %s\n", path);
        exit(1);
    }
    size_t n = fread(PW, sizeof(float), POLICY_NPARAM, f);
    fclose(f);
    if (n != POLICY_NPARAM) {
        fprintf(stderr, "policy_load: %s has %zu of %d floats\n", path, n, POLICY_NPARAM);
        exit(1);
    }
}

static void policy_save(const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "policy_save: cannot open %s\n", path);
        exit(1);
    }
    fwrite(PW, sizeof(float), POLICY_NPARAM, f);
    fclose(f);
}

static void policy_reset(PolicyState* s) {
    for (int i = 0; i < POLICY_HIDDEN; i++) s->h[i] = P_H0[i];
}

static void policy_step(PolicyState* s, const float* obs, float* action) {
    float x[POLICY_HIDDEN];
    for (int i = 0; i < POLICY_HIDDEN; i++) {
        float v = P_B0[i];
        for (int j = 0; j < RAPTOR_OBS_DIM; j++) v += P_W0[i * RAPTOR_OBS_DIM + j] * obs[j];
        x[i] = v > 0 ? v : 0;
    }

    float gi[48], gh[48];
    for (int i = 0; i < 48; i++) {
        float a = P_BI[i], b = P_BH[i];
        for (int j = 0; j < POLICY_HIDDEN; j++) {
            a += P_WI[i * POLICY_HIDDEN + j] * x[j];
            b += P_WH[i * POLICY_HIDDEN + j] * s->h[j];
        }
        gi[i] = a;
        gh[i] = b;
    }

    for (int i = 0; i < POLICY_HIDDEN; i++) {
        float r = 1.0f / (1.0f + expf(-(gi[i] + gh[i])));
        float z = 1.0f / (1.0f + expf(-(gi[POLICY_HIDDEN + i] + gh[POLICY_HIDDEN + i])));
        float n = tanhf(gi[2 * POLICY_HIDDEN + i] + r * gh[2 * POLICY_HIDDEN + i]);
        s->h[i] = (1.0f - z) * n + z * s->h[i];
    }

    for (int i = 0; i < RAPTOR_ACT_DIM; i++) {
        float v = P_B2[i];
        for (int j = 0; j < POLICY_HIDDEN; j++) v += P_W2[i * POLICY_HIDDEN + j] * s->h[j];
        action[i] = v;
    }
}
