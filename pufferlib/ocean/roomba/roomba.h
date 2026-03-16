#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define PRIMARY (Color){155, 208, 207, 255}
#define ON_PRIMARY (Color){12, 72, 72, 255}
#define PRIMARY_CONTAINER (Color){37, 90, 90, 255}
#define SURFACE (Color){10, 15, 15, 255}
#define ON_SURFACE (Color){220, 232, 232, 255}
#define ON_SURFACE_VARIANT (Color){162, 173, 173, 255}

#define PIXELS_PER_MM 1
#define ROBOT_DIAMETER (329.9f * PIXELS_PER_MM)
#define ROBOT_RADIUS (ROBOT_DIAMETER / 2.0f)
#define COVERAGE_RADIUS (90.0f * PIXELS_PER_MM)
#define GRID_COLS 12
#define GRID_ROWS 12
#define N_DOTS (GRID_COLS * GRID_ROWS)
#define N_NEAREST_DOTS 2
#define OBS_COVERAGE_OFFSET 11
#define SUCCESS_COVERAGE 0.90f
#define COLLISION_PENALTY 0.05f
#define SHAPING_WEIGHT 0.25f

Font monaspace;

typedef struct {
    float perf;
    float coverage;
    float n;
} Log;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;

    float width;
    float height;
    int speed;
    float dt;
    int tick_limit;
    int wheel_base;

    float x;
    float y;
    float bearing;
    float coverage[N_DOTS];
    int covered_count;
    int tick;
    float progress_prev;
} Roomba;

static float clampf(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static void wrap_around_angle(float* angle) {
    while (*angle < -PI) {
        *angle += 2.0f * PI;
    }
    while (*angle > PI) {
        *angle -= 2.0f * PI;
    }
}

static float dot_x(int index, float width) {
    int col = index % GRID_COLS;
    return ((float)col + 0.5f) * width / GRID_COLS;
}

static float dot_y(int index, float height) {
    int row = index / GRID_COLS;
    return ((float)row + 0.5f) * height / GRID_ROWS;
}

static float coverage_fraction(const Roomba* env) {
    return (float)env->covered_count / (float)N_DOTS;
}

static int update_coverage(Roomba* env) {
    int newly_covered = 0;

    for (int i = 0; i < N_DOTS; i++) {
        if (env->coverage[i] > 0.5f) {
            continue;
        }

        float dx = dot_x(i, env->width) - env->x;
        float dy = dot_y(i, env->height) - env->y;
        if (dx * dx + dy * dy <= COVERAGE_RADIUS * COVERAGE_RADIUS) {
            env->coverage[i] = 1.0f;
            env->covered_count += 1;
            newly_covered += 1;
        }
    }

    return newly_covered;
}

static float nearest_unvisited_distance(
    const Roomba* env,
    float* out_dx,
    float* out_dy
) {
    float best_distance_sq = FLT_MAX;
    float best_dx = 0.0f;
    float best_dy = 0.0f;

    for (int i = 0; i < N_DOTS; i++) {
        if (env->coverage[i] > 0.5f) {
            continue;
        }

        float dx = dot_x(i, env->width) - env->x;
        float dy = dot_y(i, env->height) - env->y;
        float distance_sq = dx * dx + dy * dy;
        if (distance_sq < best_distance_sq) {
            best_distance_sq = distance_sq;
            best_dx = dx;
            best_dy = dy;
        }
    }

    if (best_distance_sq == FLT_MAX) {
        *out_dx = 0.0f;
        *out_dy = 0.0f;
        return 0.0f;
    }

    *out_dx = best_dx;
    *out_dy = best_dy;
    return sqrtf(best_distance_sq);
}

static void update_obs(Roomba* env) {
    env->observations[0] = env->x / env->width;
    env->observations[1] = env->y / env->height;
    env->observations[2] = cosf(env->bearing) * 0.5f + 0.5f;
    env->observations[3] = sinf(env->bearing) * 0.5f + 0.5f;
    env->observations[4] = coverage_fraction(env);

    float best_distance_sq[N_NEAREST_DOTS];
    int best_index[N_NEAREST_DOTS];
    for (int slot = 0; slot < N_NEAREST_DOTS; slot++) {
        best_distance_sq[slot] = FLT_MAX;
        best_index[slot] = -1;
    }

    for (int i = 0; i < N_DOTS; i++) {
        if (env->coverage[i] > 0.5f) {
            continue;
        }

        float dx = dot_x(i, env->width) - env->x;
        float dy = dot_y(i, env->height) - env->y;
        float distance_sq = dx * dx + dy * dy;
        for (int slot = 0; slot < N_NEAREST_DOTS; slot++) {
            if (distance_sq < best_distance_sq[slot]) {
                for (int shift = N_NEAREST_DOTS - 1; shift > slot; shift--) {
                    best_distance_sq[shift] = best_distance_sq[shift - 1];
                    best_index[shift] = best_index[shift - 1];
                }
                best_distance_sq[slot] = distance_sq;
                best_index[slot] = i;
                break;
            }
        }
    }

    float diagonal = sqrtf(env->width * env->width + env->height * env->height);
    for (int slot = 0; slot < N_NEAREST_DOTS; slot++) {
        int obs_offset = 5 + slot * 3;
        if (best_index[slot] < 0) {
            env->observations[obs_offset + 0] = 0.0f;
            env->observations[obs_offset + 1] = 0.5f;
            env->observations[obs_offset + 2] = 0.5f;
            continue;
        }

        float dx = dot_x(best_index[slot], env->width) - env->x;
        float dy = dot_y(best_index[slot], env->height) - env->y;
        float angle = atan2f(dy, dx) - env->bearing;
        wrap_around_angle(&angle);

        env->observations[obs_offset + 0] =
            sqrtf(best_distance_sq[slot]) / diagonal;
        env->observations[obs_offset + 1] = cosf(angle) * 0.5f + 0.5f;
        env->observations[obs_offset + 2] = sinf(angle) * 0.5f + 0.5f;
    }

    for (int i = 0; i < N_DOTS; i++) {
        env->observations[OBS_COVERAGE_OFFSET + i] = env->coverage[i];
    }
}

static float get_progress(Roomba* env) {
    float dx;
    float dy;
    return -nearest_unvisited_distance(env, &dx, &dy);
}

static float get_max_progress(Roomba* env) {
    return env->speed * env->dt;
}

void c_reset(Roomba* env) {
    env->x = COVERAGE_RADIUS;
    env->y = COVERAGE_RADIUS;
    env->bearing = 0.0f;
    memset(env->coverage, 0, sizeof(env->coverage));
    env->covered_count = 0;
    env->tick = 0;

    update_coverage(env);
    env->progress_prev = get_progress(env);
    update_obs(env);
}

void c_step(Roomba* env) {
    float action_x = clampf(env->actions[0], -1.0f, 1.0f);
    float action_y = clampf(env->actions[1], -1.0f, 1.0f);
    float norm = sqrtf(action_x * action_x + action_y * action_y);
    if (norm > 1.0f) {
        action_x /= norm;
        action_y /= norm;
        norm = 1.0f;
    }

    float step = env->speed * env->dt;
    float next_x = env->x + action_x * step;
    float next_y = env->y + action_y * step;
    bool collision =
        next_x < 0.0f || next_x > env->width || next_y < 0.0f || next_y > env->height;

    env->x = clampf(next_x, 0.0f, env->width);
    env->y = clampf(next_y, 0.0f, env->height);
    if (norm > 1e-6f) {
        env->bearing = atan2f(action_y, action_x);
    }

    int newly_covered = update_coverage(env);
    float progress = get_progress(env);
    float reward = (float)newly_covered / (float)N_DOTS;
    if (newly_covered == 0 && env->covered_count < N_DOTS) {
        reward +=
            SHAPING_WEIGHT * (progress - env->progress_prev) / get_max_progress(env);
    }
    if (collision) {
        reward -= COLLISION_PENALTY;
    }

    env->tick += 1;
    bool full_coverage = env->covered_count == N_DOTS;
    bool timeout = env->tick >= env->tick_limit;
    float coverage = coverage_fraction(env);
    if (full_coverage) {
        reward += 1.0f;
    }
    env->rewards[0] = reward;

    if (full_coverage || timeout) {
        if (coverage >= SUCCESS_COVERAGE) {
            env->log.perf += 1.0f;
        }
        env->log.coverage += coverage;
        env->log.n += 1.0f;
        c_reset(env);
        env->terminals[0] = 1;
        return;
    }

    update_obs(env);
    env->progress_prev = progress;
    env->terminals[0] = 0;
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

    float x = env->observations[0] * env->width;
    float y = env->observations[1] * env->height;
    float heading_cos = env->observations[2] * 2.0f - 1.0f;
    float heading_sin = env->observations[3] * 2.0f - 1.0f;
    float bearing = atan2f(heading_sin, heading_cos);
    float coverage = env->observations[4];

    BeginDrawing();
    ClearBackground(SURFACE);

    for (int i = 0; i < N_DOTS; i++) {
        Color color =
            env->observations[OBS_COVERAGE_OFFSET + i] > 0.5f
                ? ON_SURFACE_VARIANT
                : PRIMARY_CONTAINER;
        DrawCircle(dot_x(i, env->width), dot_y(i, env->height), 5 * PIXELS_PER_MM, color);
    }

    float nearest_distance =
        env->observations[5] * sqrtf(env->width * env->width + env->height * env->height);
    float nearest_cos = env->observations[6] * 2.0f - 1.0f;
    float nearest_sin = env->observations[7] * 2.0f - 1.0f;
    float nearest_bearing = bearing + atan2f(nearest_sin, nearest_cos);
    DrawLine(
        x,
        y,
        x + nearest_distance * cosf(nearest_bearing),
        y + nearest_distance * sinf(nearest_bearing),
        PRIMARY_CONTAINER
    );

    DrawCircle(x, y, COVERAGE_RADIUS, Fade(PRIMARY_CONTAINER, 0.15f));
    DrawCircle(x, y, ROBOT_RADIUS, PRIMARY);
    DrawLine(
        x,
        y,
        x + ROBOT_RADIUS * cosf(bearing),
        y + ROBOT_RADIUS * sinf(bearing),
        ON_PRIMARY
    );

    DrawTextEx(
        monaspace,
        TextFormat("Coverage %.1f%%", coverage * 100.0f),
        (Vector2){0, 0},
        20,
        0,
        PRIMARY
    );

    EndDrawing();
}

void c_close(Roomba* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
