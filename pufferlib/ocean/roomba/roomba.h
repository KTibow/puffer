#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define PUFF_RED (Color){250, 116, 111, 255}
#define PUFF_CYAN (Color){155, 208, 207, 255}
#define PUFF_ON_CYAN (Color){12, 72, 72, 255}
#define PUFF_WHITE (Color){220, 232, 232, 241}
#define PUFF_BACKGROUND (Color){10, 15, 15, 255}
#define EPSILON_SECONDS 0.01f
#define ROBOT_DIAMETER 329.9f
#define ROBOT_RADIUS (ROBOT_DIAMETER / 2)
#define COVERAGE_RESOLUTION 20
// todo: these are starts not centers
#define COVERAGE_X(c) (env->width / COVERAGE_RESOLUTION * c)
#define COVERAGE_Y(r) (env->height / COVERAGE_RESOLUTION * r)

Font monaspace;

// Only use floats!
typedef struct {
    float perf;
    float n; // Required as the last field
} Log;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field. Ensure type matches in .py and .c
    float* actions;              // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    // config
    float width;
    float height;
    int speed;
    float dt;
    int tick_limit;
    int wheel_base;

    // state
    bool coverage[COVERAGE_RESOLUTION][COVERAGE_RESOLUTION];
    float x;
    float y;
    float bearing;
    float goalX;
    float goalY;
    float tick;
    float progress_prev;
} Roomba;

void wrap_around_angle(float *angle) {
    if (*angle < -PI) *angle += 2.0f * PI;
    if (*angle >  PI) *angle -= 2.0f * PI;
}

void update_obs(Roomba* env) {
    env->observations[0] = env->x / env->width;
    env->observations[1] = env->y / env->height;
    env->observations[2] = cosf(env->bearing) * 0.5f + 0.5f;
    env->observations[3] = sinf(env->bearing) * 0.5f + 0.5f;
    float goalDX = env->goalX - env->x;
    float goalDY = env->goalY - env->y;
    env->observations[4] = sqrtf(goalDX * goalDX + goalDY * goalDY) / sqrtf(env->width * env->width + env->height * env->height);
    float angle = atan2f(goalDX, goalDY) - env->bearing;
    wrap_around_angle(&angle);
    env->observations[5] = cosf(angle) * 0.5f + 0.5f;
    env->observations[6] = sinf(angle) * 0.5f + 0.5f;
}
float get_progress(Roomba* env) {
    float dx = env->x - env->goalX;
    float dy = env->y - env->goalY;
    return -sqrtf(dx*dx + dy*dy);
}
float get_max_progress(Roomba* env) {
    return env->speed * env->dt;
}

void c_reset(Roomba* env) {
    for (int r = 0; r < COVERAGE_RESOLUTION; r++) {
        for (int c = 0; c < COVERAGE_RESOLUTION; c++) {
            env->coverage[r][c] = false;
        }
    }
    env->x = GetRandomValue(0, env->width);
    env->y = GetRandomValue(0, env->height);
    env->bearing = 0;
    env->goalX = GetRandomValue(0, env->width);
    env->goalY = GetRandomValue(0, env->height);
    env->tick = 0;
    env->progress_prev = get_progress(env);
    update_obs(env);
}

void c_step(Roomba* env) {
    update_obs(env);

    for (int i = 0; i < (env->dt / EPSILON_SECONDS); i++) {
        float left_wheel = env->actions[0] * env->speed * EPSILON_SECONDS;
        float right_wheel = env->actions[1] * env->speed * EPSILON_SECONDS;

        float linear_displacement = (left_wheel + right_wheel) / 2.0f;
        float angular_displacement = (left_wheel - right_wheel) / env->wheel_base;

        env->x += linear_displacement * cosf(env->bearing);
        env->y += linear_displacement * sinf(env->bearing);
        env->bearing += angular_displacement;
        wrap_around_angle(&env->bearing);
        for (int r = 0; r < COVERAGE_RESOLUTION; r++) {
            for (int c = 0; c < COVERAGE_RESOLUTION; c++) {
                if (env->coverage[r][c]) continue;
                float x = COVERAGE_X(c);
                float y = COVERAGE_Y(r);
                float dx = env->x - x;
                float dy = env->y - y;
                if (sqrtf(dx * dx + dy * dy) < 10.0f) {
                    env->coverage[r][c] = true;
                }
            }
        }
    }

    float progress = get_progress(env);
    bool success = progress > -50.0f;
    bool death = env->x < 0 || env->x > env->width ||
        env->y < 0 || env-> y > env->height ||
        env->tick == env->tick_limit;

    float max_progress = get_max_progress(env);
    float reward = (progress - env->progress_prev) / max_progress * 0.5f;
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

    update_obs(env);
    env->terminals[0] = 0;
    env->progress_prev = progress;
    env->tick++;
}

void c_render(Roomba* env) {
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_WINDOW_HIGHDPI);
        InitWindow(env->width, env->height, "3omba");
        SetTargetFPS(1 / env->dt);
        monaspace = LoadFont("resources/roomba/MonaspaceNeon-Regular.otf");
    }

    if (WindowShouldClose()) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    for (int r = 0; r < COVERAGE_RESOLUTION; r++) {
        for (int c = 0; c < COVERAGE_RESOLUTION; c++) {
            if (env->coverage[r][c]) continue;
            DrawRectangle(COVERAGE_X(c), COVERAGE_Y(r), COVERAGE_X(1), COVERAGE_Y(1), PUFF_ON_CYAN);
        }
    }
    DrawCircle(env->goalX, env->goalY, 5, PUFF_RED);
    DrawCircle(env->x, env->y, ROBOT_RADIUS, PUFF_CYAN);
    DrawLine(env->x, env->y, env->x + ROBOT_RADIUS * cosf(env->bearing), env->y + ROBOT_RADIUS * sinf(env->bearing), PUFF_ON_CYAN);
    DrawTextEx(monaspace, TextFormat("L%+.2f R%+.2f", env->actions[0], env->actions[1]), (Vector2){0,0}, 20, 0, PUFF_CYAN);

    EndDrawing();
}

void c_close(Roomba* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
