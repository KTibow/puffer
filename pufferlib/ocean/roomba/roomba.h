#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"

#define PUFF_RED (Color){187, 0, 0, 255}
#define PUFF_CYAN (Color){0, 187, 187, 255}
#define PUFF_WHITE (Color){241, 241, 241, 241}
#define PUFF_BACKGROUND (Color){6, 24, 24, 255}

#define PIXELS_PER_MM 1
#define COVERAGE_DOTS_SIZE 20

#define ROBOT_DIAMETER (329.9f * PIXELS_PER_MM)
#define ROBOT_RADIUS (ROBOT_DIAMETER / 2)

Font monaspace;

// Only use floats!
typedef struct {
    float suicide; // % runs into wall
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
    int wheel_base;
    int brush_length;

    // state
    float x;
    float y;
    float bearing;
    int tick;
    int last_coverage_tick;
    bool coverage_dots[COVERAGE_DOTS_SIZE][COVERAGE_DOTS_SIZE];
    float progress_prev;
} Roomba;

void update_obs(Roomba* env) {
    env->observations[0] = env->x / env->width;
    env->observations[1] = env->y / env->height;
    env->observations[2] = cosf(env->bearing) * 0.5f + 0.5f;
    env->observations[3] = sinf(env->bearing) * 0.5f + 0.5f;
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

// Credit: Gemini 3.1 Pro
void update_progress(float old_x, float old_y, float old_bearing, Roomba* env) {
    float cell_w = env->width / COVERAGE_DOTS_SIZE;
    float cell_h = env->height / COVERAGE_DOTS_SIZE;

    // Treat the brush as a circle for omnidirectional sweeping.
    float radius = env->brush_length / 2.0f;
    float radius_sq = radius * radius;

    Vector2 start = { old_x, old_y };
    Vector2 end = { env->x, env->y };

    // 1. Calculate bounding box of the movement, padded by the brush radius
    float min_x = fminf(start.x, end.x) - radius;
    float max_x = fmaxf(start.x, end.x) + radius;
    float min_y = fminf(start.y, end.y) - radius;
    float max_y = fmaxf(start.y, end.y) + radius;

    // 2. Convert to grid bounds and clamp to prevent array out-of-bounds
    int min_c = (int)fmaxf(0.0f, min_x / cell_w);
    int max_c = (int)fminf(COVERAGE_DOTS_SIZE - 1.0f, max_x / cell_w);
    int min_r = (int)fmaxf(0.0f, min_y / cell_h);
    int max_r = (int)fminf(COVERAGE_DOTS_SIZE - 1.0f, max_y / cell_h);

    // Precalculate squared segment length to avoid square roots
    float segment_length_sq = Vector2DistanceSqr(start, end);

    // 3. Loop over the bounding box
    for (int r = min_r; r <= max_r; r++) {
        float cy = cell_h * r + cell_h * 0.5f;

        for (int c = min_c; c <= max_c; c++) {
            if (env->coverage_dots[r][c]) continue; // Already clean

            Vector2 cell_center = { cell_w * c + cell_w * 0.5f, cy };
            float dist_sq;

            // 4. Find the shortest distance from the cell center to the movement line segment
            if (segment_length_sq == 0.0f) {
                // Robot didn't move, just check distance to start point
                dist_sq = Vector2DistanceSqr(cell_center, start);
            } else {
                // Project the cell center onto the line segment
                Vector2 diff = Vector2Subtract(cell_center, start);
                Vector2 dir = Vector2Subtract(end, start);

                // Calculate interpolation factor 't' along the segment
                float t = (diff.x * dir.x + diff.y * dir.y) / segment_length_sq;

                // Clamp 't' between 0 and 1 so we don't check infinitely past the segment
                t = fmaxf(0.0f, fminf(1.0f, t));

                // Find the closest point on the line segment
                Vector2 closest_point = { start.x + t * dir.x, start.y + t * dir.y };
                dist_sq = Vector2DistanceSqr(cell_center, closest_point);
            }

            // 5. If the cell center is within the brush radius, it's covered!
            if (dist_sq <= radius_sq) {
                env->coverage_dots[r][c] = true;
                env->last_coverage_tick = env->tick;
            }
        }
    }
}

int get_progress(Roomba* env) {
    int progress = 0;
    for (size_t r = 0; r < COVERAGE_DOTS_SIZE; r++) {
        for (size_t c = 0; c < COVERAGE_DOTS_SIZE; c++) {
            if (env->coverage_dots[r][c]) {
                progress++;
            }
        }
    }
    return progress;
}
int get_max_progress() {
    return 5;
}
int get_max_coverage() {
    return COVERAGE_DOTS_SIZE * COVERAGE_DOTS_SIZE;
}

void c_reset(Roomba* env) {
    env->x = 20;
    env->y = 20;
    env->bearing = PI/2;
    // env->x = GetRandomValue(0, env->width);
    // env->y = GetRandomValue(0, env->height);
    // env->bearing = ((float)GetRandomValue(0, 1000) / 1000.0f) * 2.0f * PI;
    env->last_coverage_tick = 0;
    env->tick = 0;
    memset(env->coverage_dots, 0, sizeof env->coverage_dots);
    env->progress_prev = get_progress(env);
    update_obs(env);
}

void c_step(Roomba* env) {
    update_obs(env);

    float old_x = env->x;
    float old_y = env->y;
    float old_bearing = env->bearing;

    // todo switch to [forward, rotation]?
    float left_wheel = env->actions[0] * env->speed * env->dt;
    float right_wheel = env->actions[1] * env->speed * env->dt;
    update_pose(&env->x, &env->y, &env->bearing, left_wheel, right_wheel, env->wheel_base);

    update_progress(old_x, old_y, old_bearing, env);
    float progress = get_progress(env);
    bool end = env->tick - env->last_coverage_tick > 30;
    bool suicide = env->x < 0 || env->x > env->width ||
        env->y < 0 || env-> y > env->height;

    float max_progress = get_max_progress();
    float max_coverage = get_max_coverage();
    float reward = (progress - env->progress_prev) / max_progress * 0.5f;
    if (end) {
        float term = progress / max_coverage - 0.9f;
        term *= 10.0f;
        if (term < -1.0f) {
            term = -1.0f;
        }
        reward += term;
    }
    if (suicide) {
        reward -= 1.0f;
        env->log.suicide += 1;
    }
    reward -= 0.01f; // slightly incentivize speed
    env->rewards[0] = reward;

    if (end || suicide) {
        env->log.coverage = progress / max_coverage;
        env->log.n += 1;
        env->terminals[0] = 1;
        c_reset(env);
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

    float cell_w = env->width / COVERAGE_DOTS_SIZE;
    float cell_h = env->height / COVERAGE_DOTS_SIZE;
    for (size_t r = 0; r < COVERAGE_DOTS_SIZE; r++) {
        for (size_t c = 0; c < COVERAGE_DOTS_SIZE; c++) {
            float x = cell_w * c + cell_w * 0.5f;
            float y = cell_h * r + cell_h * 0.5f;
            if (!env->coverage_dots[r][c]) {
                DrawCircle(x, y, 10 * PIXELS_PER_MM, PUFF_RED);
            }
        }
    }

    DrawCircle(env->x, env->y, ROBOT_RADIUS, PUFF_CYAN);
    DrawLine(env->x, env->y, env->x + ROBOT_RADIUS * cosf(env->bearing), env->y + ROBOT_RADIUS * sinf(env->bearing), PUFF_WHITE);
    DrawTextEx(monaspace, TextFormat("L%+.2f R%+.2f T%4d P%d", env->actions[0], env->actions[1], env->tick, get_progress(env)), (Vector2){0,0}, 20, 0, PUFF_CYAN);

    EndDrawing();
}

void c_close(Roomba* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
