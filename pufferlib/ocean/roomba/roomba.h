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
#define ROBOT_DIAMETER (329.9f * PIXELS_PER_MM)
#define ROBOT_RADIUS (ROBOT_DIAMETER / 2)

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
    float goalX;
    float goalY;
    float tick;
    float progress_prev;
} Roomba;

void update_obs(Roomba* env) {
    env->observations[0] = env->x / env->width;
    env->observations[1] = env->y / env->height;
    env->observations[2] = cosf(env->bearing) * 0.5f + 0.5f;
    env->observations[3] = sinf(env->bearing) * 0.5f + 0.5f;
    env->observations[4] = env->goalX / env->width;
    env->observations[5] = env->goalY / env->height;
}
float get_progress(Roomba* env) {
    float dx = env->x - env->goalX;
    float dy = env->y - env->goalY;
    return -sqrtf(dx*dx + dy*dy);
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
    if (*bearing >  PI) *bearing -= 2.0f * PI;
    if (*bearing < -PI) *bearing += 2.0f * PI;
}

void c_reset(Roomba* env) {
    env->x = env->width / 2;
    env->y = env->height / 2;
    env->bearing = 0;
    env->goalX = GetRandomValue(0, env->width);
    env->goalY = GetRandomValue(0, env->height);
    env->tick = 0;
    env->progress_prev = get_progress(env);
    update_obs(env);
}

void c_step(Roomba* env) {
    update_obs(env);

    float left_wheel = env->actions[0] * env->speed * env->dt;
    float right_wheel = env->actions[1] * env->speed * env->dt;
    update_pose(&env->x, &env->y, &env->bearing, left_wheel, right_wheel, env->wheel_base);

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
        InitWindow(env->width * PIXELS_PER_MM, env->height * PIXELS_PER_MM, "3omba");
        SetTargetFPS(1 / env->dt);
        monaspace = LoadFont("resources/roomba/MonaspaceNeon-Regular.otf");
    }

    if (WindowShouldClose()) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    DrawCircleLines(env->goalX, env->goalY, ROBOT_RADIUS, PUFF_CYAN);
    DrawCircle(env->x, env->y, ROBOT_RADIUS, PUFF_CYAN);
    DrawLine(env->x, env->y, env->x + ROBOT_RADIUS * cosf(env->bearing), env->y + ROBOT_RADIUS * sinf(env->bearing), PUFF_WHITE);
    DrawTextEx(monaspace, TextFormat("L%+.2f R%+.2f", env->actions[0], env->actions[1]), (Vector2){0,0}, 20, 0, PUFF_CYAN);

    EndDrawing();
}

void c_close(Roomba* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
