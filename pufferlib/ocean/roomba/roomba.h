#include <float.h>
#include <math.h>
#include <stdbool.h>
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
#define COLLISION_PENALTY 0.02f
#define SHAPING_WEIGHT 0.35f
#define ALIGNMENT_SHAPING_WEIGHT 0.15f

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
    int diverse_resets;

    float x;
    float y;
    float bearing;
    float coverage[N_DOTS];
    int covered_count;
    int tick;
    float progress_prev;
    float alignment_prev;
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

static float randf(float min_value, float max_value) {
    return min_value + ((float)rand() / (float)RAND_MAX) * (max_value - min_value);
}

static int randi(int min_value, int max_value) {
    if (max_value <= min_value) {
        return min_value;
    }
    return min_value + rand() % (max_value - min_value + 1);
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

static int boustrophedon_index(int order_idx) {
    int row = order_idx / GRID_COLS;
    int offset = order_idx % GRID_COLS;
    int col = (row % 2 == 0) ? offset : (GRID_COLS - 1 - offset);
    return row * GRID_COLS + col;
}

static float coverage_fraction(const Roomba* env) {
    return (float)env->covered_count / (float)N_DOTS;
}

static void clear_coverage(Roomba* env, float value) {
    for (int i = 0; i < N_DOTS; i++) {
        env->coverage[i] = value;
    }
}

static void recount_coverage(Roomba* env) {
    int count = 0;
    for (int i = 0; i < N_DOTS; i++) {
        if (env->coverage[i] > 0.5f) {
            count += 1;
        }
    }
    env->covered_count = count;
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

static float closest_unvisited_distance_to_pose(const Roomba* env, float x, float y) {
    float best_distance_sq = FLT_MAX;

    for (int i = 0; i < N_DOTS; i++) {
        if (env->coverage[i] > 0.5f) {
            continue;
        }

        float dx = dot_x(i, env->width) - x;
        float dy = dot_y(i, env->height) - y;
        float distance_sq = dx * dx + dy * dy;
        if (distance_sq < best_distance_sq) {
            best_distance_sq = distance_sq;
        }
    }

    if (best_distance_sq == FLT_MAX) {
        return 0.0f;
    }

    return sqrtf(best_distance_sq);
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

static void set_pose(Roomba* env, float x, float y, float bearing) {
    env->x = x;
    env->y = y;
    env->bearing = bearing;
}

static void random_pose_heading(Roomba* env, bool prefer_wall, bool avoid_unvisited) {
    float min_x = COVERAGE_RADIUS;
    float max_x = env->width - COVERAGE_RADIUS;
    float min_y = COVERAGE_RADIUS;
    float max_y = env->height - COVERAGE_RADIUS;

    float x = min_x;
    float y = min_y;
    for (int attempt = 0; attempt < 64; attempt++) {
        if (prefer_wall && rand() % 2 == 0) {
            float band = COVERAGE_RADIUS * 1.35f;
            int wall = rand() % 4;
            if (wall == 0) {
                x = randf(min_x, min_x + band);
                y = randf(min_y, max_y);
            } else if (wall == 1) {
                x = randf(max_x - band, max_x);
                y = randf(min_y, max_y);
            } else if (wall == 2) {
                x = randf(min_x, max_x);
                y = randf(min_y, min_y + band);
            } else {
                x = randf(min_x, max_x);
                y = randf(max_y - band, max_y);
            }
        } else {
            x = randf(min_x, max_x);
            y = randf(min_y, max_y);
        }

        if (!avoid_unvisited ||
            closest_unvisited_distance_to_pose(env, x, y) > COVERAGE_RADIUS * 1.15f) {
            break;
        }
    }

    set_pose(env, x, y, randf(-PI, PI));
}

static void setup_fresh_reset(Roomba* env) {
    clear_coverage(env, 0.0f);
    recount_coverage(env);
    set_pose(env, COVERAGE_RADIUS, COVERAGE_RADIUS, 0.0f);
}

static void setup_prefix_reset(Roomba* env) {
    clear_coverage(env, 0.0f);
    int covered_target = randi(N_DOTS / 5, (N_DOTS * 4) / 5);
    for (int order_idx = 0; order_idx < covered_target; order_idx++) {
        env->coverage[boustrophedon_index(order_idx)] = 1.0f;
    }
    recount_coverage(env);

    int frontier_order = covered_target;
    if (frontier_order < 0) {
        frontier_order = 0;
    }
    if (frontier_order > N_DOTS - 1) {
        frontier_order = N_DOTS - 1;
    }
    int frontier_idx = boustrophedon_index(frontier_order);
    int frontier_row = frontier_idx / GRID_COLS;
    int direction = (frontier_row % 2 == 0) ? 1 : -1;
    float fx = dot_x(frontier_idx, env->width);
    float fy = dot_y(frontier_idx, env->height);
    float x = clampf(
        fx - direction * COVERAGE_RADIUS * 1.3f,
        COVERAGE_RADIUS,
        env->width - COVERAGE_RADIUS
    );
    float y = clampf(
        fy + randf(-0.35f, 0.35f) * COVERAGE_RADIUS,
        COVERAGE_RADIUS,
        env->height - COVERAGE_RADIUS
    );
    float bearing = (direction > 0 ? 0.0f : PI) + randf(-0.25f, 0.25f);
    set_pose(env, x, y, bearing);
}

static void shuffle_indices(int* indices, int n) {
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

static void setup_sparse_remaining_reset(Roomba* env, int min_remaining, int max_remaining) {
    clear_coverage(env, 1.0f);
    int remaining = randi(min_remaining, max_remaining);
    int indices[N_DOTS];
    shuffle_indices(indices, N_DOTS);
    for (int i = 0; i < remaining; i++) {
        env->coverage[indices[i]] = 0.0f;
    }
    recount_coverage(env);
    random_pose_heading(env, true, true);
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

static float get_alignment(Roomba* env) {
    float dx;
    float dy;
    float distance = nearest_unvisited_distance(env, &dx, &dy);
    if (distance <= 1e-6f) {
        return 1.0f;
    }

    float angle = atan2f(dy, dx) - env->bearing;
    wrap_around_angle(&angle);
    return cosf(angle);
}

static float get_max_progress(Roomba* env) {
    return env->speed * env->dt;
}

static void update_pose(
    float* x,
    float* y,
    float* bearing,
    float d_l,
    float d_r,
    float wheel_base
) {
    float v = (d_r + d_l) / 2.0f;
    float w = (d_l - d_r) / wheel_base;

    if (fabsf(w) < 1e-6f) {
        *x += v * cosf(*bearing);
        *y += v * sinf(*bearing);
    } else {
        float theta_new = *bearing + w;
        *x += (v / w) * (sinf(theta_new) - sinf(*bearing));
        *y -= (v / w) * (cosf(theta_new) - cosf(*bearing));
        *bearing = theta_new;
    }

    wrap_around_angle(bearing);
}

void c_reset(Roomba* env) {
    int mode = 0;
    if (env->diverse_resets) {
        int draw = rand() % 100;
        if (draw < 50) {
            mode = 0;
        } else if (draw < 85) {
            mode = 1;
        } else if (draw < 95) {
            mode = 2;
        } else {
            mode = 3;
        }
    }

    if (mode == 0) {
        setup_fresh_reset(env);
    } else if (mode == 1) {
        setup_prefix_reset(env);
    } else if (mode == 2) {
        setup_sparse_remaining_reset(env, 2, 6);
    } else {
        setup_sparse_remaining_reset(env, 1, 1);
    }

    env->tick = 0;
    update_coverage(env);
    env->progress_prev = get_progress(env);
    env->alignment_prev = get_alignment(env);
    update_obs(env);
}

void c_step(Roomba* env) {
    float left_wheel = clampf(env->actions[0], -1.0f, 1.0f) * env->speed * env->dt;
    float right_wheel = clampf(env->actions[1], -1.0f, 1.0f) * env->speed * env->dt;

    update_pose(&env->x, &env->y, &env->bearing, left_wheel, right_wheel, env->wheel_base);

    float min_x = COVERAGE_RADIUS;
    float max_x = env->width - COVERAGE_RADIUS;
    float min_y = COVERAGE_RADIUS;
    float max_y = env->height - COVERAGE_RADIUS;
    bool collision =
        env->x < min_x || env->x > max_x || env->y < min_y || env->y > max_y;
    env->x = clampf(env->x, min_x, max_x);
    env->y = clampf(env->y, min_y, max_y);

    int newly_covered = update_coverage(env);
    float progress = get_progress(env);
    float alignment = get_alignment(env);
    float reward = (float)newly_covered / (float)N_DOTS;
    if (newly_covered == 0 && env->covered_count < N_DOTS) {
        reward +=
            SHAPING_WEIGHT * (progress - env->progress_prev) / get_max_progress(env);
    }
    if (env->covered_count < N_DOTS) {
        reward += ALIGNMENT_SHAPING_WEIGHT * (alignment - env->alignment_prev);
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
    env->alignment_prev = alignment;
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
