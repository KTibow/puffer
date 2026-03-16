#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

// #define ERROR (Color){250, 116, 111, 255}
#define PRIMARY (Color){155, 208, 207, 255}
#define ON_PRIMARY (Color){12, 72, 72, 255}
#define PRIMARY_CONTAINER (Color){37, 90, 90, 255}
#define SURFACE (Color){10, 15, 15, 255}
#define ON_SURFACE (Color){220, 232, 232, 255}
#define ON_SURFACE_VARIANT (Color){162, 173, 173, 255}
#define PIXELS_PER_MM 1
#define ROBOT_DIAMETER (329.9f * PIXELS_PER_MM)
#define ROBOT_RADIUS (ROBOT_DIAMETER / 2)
#define N_GOALS 4
#define GOAL_REACHED_DISTANCE (50.0f * PIXELS_PER_MM)
#define GOAL_REWARD (1.0f / N_GOALS)
#define STEP_PENALTY 0.0f

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
    float x;
    float y;
    float bearing;
    float goalsX[N_GOALS];
    float goalsY[N_GOALS];
    float goalsTerminations[N_GOALS];
    float tick;
    float progress_prev;
} Roomba;

void wrap_around_angle(float *angle) {
    if (*angle < -PI) *angle += 2.0f * PI;
    if (*angle >  PI) *angle -= 2.0f * PI;
}

float get_goal_distance(Roomba* env, int goal_idx) {
    float dx = env->x - env->goalsX[goal_idx];
    float dy = env->y - env->goalsY[goal_idx];
    return sqrtf(dx * dx + dy * dy);
}

bool all_goals_completed(Roomba* env) {
    for (int i = 0; i < N_GOALS; i++) {
        if (!env->goalsTerminations[i]) {
            return false;
        }
    }

    return true;
}

int collect_reached_goals(Roomba* env) {
    int goals_collected = 0;

    for (int i = 0; i < N_GOALS; i++) {
        if (env->goalsTerminations[i]) {
            continue;
        }

        if (get_goal_distance(env, i) <= GOAL_REACHED_DISTANCE) {
            env->goalsTerminations[i] = 1;
            goals_collected++;
        }
    }

    return goals_collected;
}

void update_obs(Roomba* env) {
    env->observations[0] = env->x / env->width;
    env->observations[1] = env->y / env->height;
    env->observations[2] = cosf(env->bearing) * 0.5f + 0.5f;
    env->observations[3] = sinf(env->bearing) * 0.5f + 0.5f;

    for (int i = 0; i < N_GOALS; i++) {
        if (env->goalsTerminations[i]) {
            env->observations[3 + i * 4 + 1] = 0.0f;
            env->observations[3 + i * 4 + 2] = 0.5f;
            env->observations[3 + i * 4 + 3] = 0.5f;
            env->observations[3 + i * 4 + 4] = 1.0f;
            continue;
        }

        float goalDX = env->goalsX[i] - env->x;
        float goalDY = env->goalsY[i] - env->y;
        env->observations[3 + i * 4 + 1] =
            sqrtf(goalDX * goalDX + goalDY * goalDY) /
            sqrtf(env->width * env->width + env->height * env->height);

        float angle = atan2f(goalDY, goalDX) - env->bearing;
        wrap_around_angle(&angle);
        env->observations[3 + i * 4 + 2] = cosf(angle) * 0.5f + 0.5f;
        env->observations[3 + i * 4 + 3] = sinf(angle) * 0.5f + 0.5f;
        env->observations[3 + i * 4 + 4] = 0.0f;
    }
}

float get_progress(Roomba* env) {
    float minDistance = FLT_MAX;

    for (int i = 0; i < N_GOALS; i++) {
        if (env->goalsTerminations[i]) {
            continue;
        }

        float distance = get_goal_distance(env, i);
        if (distance < minDistance) {
            minDistance = distance;
        }
    }

    if (minDistance == FLT_MAX) {
        return 0.0f;
    }

    return -minDistance;
}

float get_max_progress(Roomba* env) {
    return env->speed * env->dt;
}

// Credit: Gemini 3 Flash
void update_pose(float *x, float *y, float *bearing,
                 float d_l, float d_r,
                 float wheel_base)
{
    // 1. Calculate linear displacement and angular displacement
    float v = (d_r + d_l) / 2.0f;
    // (d_l - d_r) makes faster left wheel = positive change in bearing = Clockwise turn
    float w = (d_l - d_r) / wheel_base;

    // 2. Update position
    if (fabsf(w) < 1e-6f) {
        // Straight line (prevents division by zero)
        *x += v * cosf(*bearing);
        *y += v * sinf(*bearing);
    } else {
        // Precise arc
        float theta_new = *bearing + w;

        // In Y-down, x = ∫ v cos(θ) dt  and y = ∫ v sin(θ) dt
        // Integrating these gives:
        *x += (v / w) * (sinf(theta_new) - sinf(*bearing));
        *y -= (v / w) * (cosf(theta_new) - cosf(*bearing)); // Note the minus sign

        *bearing = theta_new;
    }

    // 3. Keep angle between -PI and PI
    wrap_around_angle(bearing);
}

void c_reset(Roomba* env) {
    env->x = GetRandomValue(0, env->width);
    env->y = GetRandomValue(0, env->height);
    env->bearing = 0;
    for (int i = 0; i < N_GOALS; i++) {
        env->goalsX[i] = GetRandomValue(0, env->width);
        env->goalsY[i] = GetRandomValue(0, env->height);
        env->goalsTerminations[i] = 0;
    }
    env->tick = 0;
    env->progress_prev = get_progress(env);
    update_obs(env);
}

void c_step(Roomba* env) {
    float left_wheel = env->actions[0] * env->speed * env->dt;
    float right_wheel = env->actions[1] * env->speed * env->dt;
    update_pose(&env->x, &env->y, &env->bearing, left_wheel, right_wheel, env->wheel_base);

    int goals_collected = collect_reached_goals(env);
    float progress = get_progress(env);
    bool success = all_goals_completed(env);
    bool death = env->x < 0 || env->x > env->width ||
        env->y < 0 || env-> y > env->height ||
        env->tick == env->tick_limit;

    float reward = goals_collected * GOAL_REWARD;
    if (goals_collected == 0) {
        float max_progress = get_max_progress(env);
        reward += (progress - env->progress_prev) / max_progress * 0.5f;
    }
    if (success) {
        env->log.perf += 1;
    }
    if (death) {
        reward -= 1.0f;
    }
    reward -= STEP_PENALTY;
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
        InitWindow(env->width * PIXELS_PER_MM, env->height * PIXELS_PER_MM, "3omba");
        SetTargetFPS(1 / env->dt);
        monaspace = LoadFont("resources/roomba/MonaspaceNeon-Regular.otf");
    }

    if (WindowShouldClose()) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(SURFACE);

    DrawCircle(env->x, env->y, ROBOT_RADIUS, PRIMARY);
    DrawLine(env->x, env->y, env->x + ROBOT_RADIUS * cosf(env->bearing), env->y + ROBOT_RADIUS * sinf(env->bearing), ON_PRIMARY);
    for (int i = 0; i < N_GOALS; i++) {
        DrawCircle(env->goalsX[i], env->goalsY[i], 5 * PIXELS_PER_MM, env->goalsTerminations[i] ? ON_SURFACE_VARIANT : PRIMARY_CONTAINER);
    }
    DrawTextEx(monaspace, TextFormat("L%+.2f R%+.2f", env->actions[0], env->actions[1]), (Vector2){0, 0}, 20, 0, PRIMARY);

    EndDrawing();
}

void c_close(Roomba* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
