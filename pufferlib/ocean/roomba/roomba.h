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
#define EPSILON_SECONDS 0.02f
#define ROBOT_DIAMETER 329.9f
#define ROBOT_RADIUS (ROBOT_DIAMETER / 2)
#define COVERAGE_RESOLUTION 20
#define X_START_COVERAGE_RECT(c) (env->width / COVERAGE_RESOLUTION * c)
#define Y_START_COVERAGE_RECT(r) (env->height / COVERAGE_RESOLUTION * r)
#define WIDTH_COVERAGE_RECT() (env->width / COVERAGE_RESOLUTION)
#define HEIGHT_COVERAGE_RECT() (env->height / COVERAGE_RESOLUTION)
#define X_COVERAGE_DOT(c) (env->width / COVERAGE_RESOLUTION * (c + 0.5f))
#define Y_COVERAGE_DOT(r) (env->height / COVERAGE_RESOLUTION * (r + 0.5f))

Font monaspace;

// Only use floats!
typedef struct {
    float perf;
    float coverage;
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
    int tick;
    int progress_prev;
} Roomba;

void wrap_around_angle(float *angle) {
    if (*angle < -PI) *angle += 2.0f * PI;
    if (*angle >  PI) *angle -= 2.0f * PI;
}
bool is_coverage_position_in_bounds(int position) {
    return position >= 0 && position < COVERAGE_RESOLUTION;
}
void conditionally_cover(Roomba* env, int r, int c) {
    if (!is_coverage_position_in_bounds(r)) return;
    if (!is_coverage_position_in_bounds(c)) return;
    env->coverage[r][c] = true;
}

int calculate_bumper_distance(Roomba* env, float relative_angle) {
    int distance = 0;
    while (distance < 50) {
        float x = env->x + cosf(env->bearing + relative_angle) * (distance + ROBOT_RADIUS);
        float y = env->y + sinf(env->bearing + relative_angle) * (distance + ROBOT_RADIUS);
        if (x < 0) break;
        if (y < 0) break;
        if (x >= env->width) break;
        if (y >= env->height) break;
        distance++;
    }
    return distance;
}

void update_obs(Roomba* env, int progress) {
    env->observations[0] = (float)calculate_bumper_distance(env, -0.22f*PI) / 50; // left bumper
    env->observations[1] = (float)calculate_bumper_distance(env, 0) / 50;
    env->observations[2] = (float)calculate_bumper_distance(env, 0.35f*PI) / 50; // right bumper
    env->observations[3] = (float)progress / (COVERAGE_RESOLUTION*COVERAGE_RESOLUTION);
}
int get_progress(Roomba* env) {
    int progress = 0;
    for (int r = 0; r < COVERAGE_RESOLUTION; r++) {
        for (int c = 0; c < COVERAGE_RESOLUTION; c++) {
            if (env->coverage[r][c]) progress++;
        }
    }
    return progress;
}

void c_reset(Roomba* env) {
    for (int r = 0; r < COVERAGE_RESOLUTION; r++) {
        for (int c = 0; c < COVERAGE_RESOLUTION; c++) {
            env->coverage[r][c] = false;
        }
    }
    env->x = GetRandomValue(ROBOT_RADIUS, env->width - ROBOT_RADIUS);
    env->y = GetRandomValue(ROBOT_RADIUS, env->height - ROBOT_RADIUS);
    env->bearing = 0;
    env->tick = 0;
    int progress = get_progress(env);
    env->progress_prev = progress;
    update_obs(env, progress);
}

void c_step(Roomba* env) {
    float reward = -0.01f; // incentivize speed

    for (int i = 0; i < (env->dt / EPSILON_SECONDS); i++) {
        float left_wheel = env->actions[0] * env->speed * EPSILON_SECONDS;
        float right_wheel = env->actions[1] * env->speed * EPSILON_SECONDS;

        float linear_displacement = (left_wheel + right_wheel) / 2.0f;
        float angular_displacement = (left_wheel - right_wheel) / env->wheel_base;

        env->x += linear_displacement * cosf(env->bearing);
        env->y += linear_displacement * sinf(env->bearing);
        env->bearing += angular_displacement;
        wrap_around_angle(&env->bearing);

        int center_r = roundf(COVERAGE_RESOLUTION * (env->y/env->height));
        int center_c = roundf(COVERAGE_RESOLUTION * (env->x/env->width));
        conditionally_cover(env, center_r, center_c);
        conditionally_cover(env, center_r - 3, center_c);
        conditionally_cover(env, center_r - 2, center_c);
        conditionally_cover(env, center_r - 1, center_c);
        conditionally_cover(env, center_r + 1, center_c);
        conditionally_cover(env, center_r + 2, center_c);
        conditionally_cover(env, center_r + 3, center_c);
        conditionally_cover(env, center_r, center_c - 3);
        conditionally_cover(env, center_r, center_c - 2);
        conditionally_cover(env, center_r, center_c - 1);
        conditionally_cover(env, center_r, center_c + 1);
        conditionally_cover(env, center_r, center_c + 2);
        conditionally_cover(env, center_r, center_c + 3);
        conditionally_cover(env, center_r - 1, center_c - 1);
        conditionally_cover(env, center_r - 1, center_c + 1);
        conditionally_cover(env, center_r + 1, center_c - 1);
        conditionally_cover(env, center_r + 1, center_c + 1);
    }

    if (env->x < ROBOT_RADIUS) {
        env->x = ROBOT_RADIUS;
        reward -= 0.1f;
    }
    if (env->x > (env->width - ROBOT_RADIUS)) {
        env->x = env->width - ROBOT_RADIUS;
        reward -= 0.1f;
    }
    if (env->y < ROBOT_RADIUS) {
        env->y = ROBOT_RADIUS;
        reward -= 0.1f;
    }
    if (env->y > (env->height - ROBOT_RADIUS)) {
        env->y = env->height - ROBOT_RADIUS;
        reward -= 0.1f;
    }

    int progress = get_progress(env);
    int max_progress = 5;
    reward += (float)(progress - env->progress_prev) / max_progress * 0.5f;

    bool success = progress > COVERAGE_RESOLUTION*COVERAGE_RESOLUTION*0.9f;
    bool death = env->tick == env->tick_limit;
    if (success) {
        reward += 1.0f;
        env->log.perf += 1;
    }
    if (death) {
        reward -= 1.0f;
    }
    env->rewards[0] = reward;

    if (success || death) {
        c_reset(env);
        env->terminals[0] = 1;
        env->log.coverage = (float)progress / (COVERAGE_RESOLUTION * COVERAGE_RESOLUTION);
        env->log.n += 1;
        return;
    }

    update_obs(env, progress);
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

    float width = WIDTH_COVERAGE_RECT();
    float height = HEIGHT_COVERAGE_RECT();
    for (int r = 0; r < COVERAGE_RESOLUTION; r++) {
        float y_start = Y_START_COVERAGE_RECT(r);
        for (int c = 0; c < COVERAGE_RESOLUTION; c++) {
            if (env->coverage[r][c]) continue;
            DrawRectangle(X_START_COVERAGE_RECT(c), y_start, width, height, PUFF_ON_CYAN);
        }
    }
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
