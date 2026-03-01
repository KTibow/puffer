#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define PUFF_RED (Color){187, 0, 0, 255}
#define PUFF_CYAN (Color){0, 187, 187, 255}
#define PUFF_WHITE (Color){241, 241, 241, 241}
#define PUFF_BACKGROUND (Color){6, 24, 24, 255}
#define PIXELS_PER_MM 1

// Only use floats!
typedef struct {
    float perf;
    float n; // Required as the last field
} Log;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field. Ensure type matches in .py and .c
    float* actions;                // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    // config
    float width;
    float height;
    int speed;
    float dt;
    int tick_limit;

    // state
    float x;
    float y;
    float goalX;
    float goalY;
    float tick;
    float progress_prev;
} Roomba;

void updateObs(Roomba* env) {
    env->observations[0] = env->x / env->width;
    env->observations[1] = env->y / env->height;
    env->observations[2] = env->goalX / env->width;
    env->observations[3] = env->goalY / env->height;
}
float getProgress(Roomba* env) {
    float dx = env->x - env->goalX;
    float dy = env->y - env->goalY;
    return -sqrtf(dx*dx + dy*dy);
}

void c_reset(Roomba* env) {
    env->x = env->width / 2;
    env->y = env->height / 2;
    env->goalX = GetRandomValue(0, env->width);
    env->goalY = GetRandomValue(0, env->height);
    env->tick = 0;
    env->progress_prev = getProgress(env);
    updateObs(env);
}

void c_step(Roomba* env) {
    updateObs(env);

    env->x += env->actions[0] * env->speed * env->dt;
    env->y += env->actions[1] * env->speed * env->dt;
    float progress = getProgress(env);
    bool success = progress > -5.0f;
    bool death = env->x < 0 || env->x > env->width ||
        env->y < 0 || env-> y > env->height ||
        env->tick == env->tick_limit;

    float reward = (progress - env->progress_prev) / env->dt * 0.05f;
    if (success) {
        reward += 1.0f;
        env->log.perf += 1;
    }
    if (death) {
        reward -= 1.0f;
    }
    reward -= 0.1f; // incentivize speed
    env->rewards[0] = reward;

    if (success || death) {
        c_reset(env);
        env->terminals[0] = 1;
        env->log.n += 1;
        return;
    }

    updateObs(env);
    env->terminals[0] = 0;
    env->progress_prev = progress;
    env->tick++;
}

void c_render(Roomba* env) {
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_WINDOW_HIGHDPI);
        InitWindow(env->width * PIXELS_PER_MM, env->height * PIXELS_PER_MM, "3omba");
        SetTargetFPS(1 / env->dt);
    }

    if (WindowShouldClose()) {
        exit(0);
    }

    DrawCircleLines(env->goalX, env->goalY, 20 * PIXELS_PER_MM, PUFF_CYAN);
    DrawCircle(env->x, env->y, 20 * PIXELS_PER_MM, PUFF_CYAN);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    EndDrawing();
}

void c_close(Roomba* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
