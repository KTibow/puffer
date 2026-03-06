#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define PUFF_RED (Color){187, 0, 0, 255}
#define PUFF_CYAN (Color){0, 187, 187, 255}
#define PUFF_WHITE (Color){241, 241, 241, 241}
#define PUFF_BACKGROUND (Color){6, 24, 24, 255}
#define SIZE 10

// Only use floats!
typedef struct {
    float score;
    float n; // Required as the last field
} Log;

typedef struct {
    Log log;                     // Required field
    unsigned char* observations; // Required field. Ensure type matches in .py and .c
    int* actions;                // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    int x;
    int y;
    int dx;
    int dy;
    bool coverage[SIZE][SIZE];
} Boustrophedon;

void update_obs(Boustrophedon* env) {
    env->observations[0] = env->y == 0;
    env->observations[1] = env->y == SIZE - 1;
    env->observations[2] = env->dx;
    env->observations[3] = env->dy;
}

void c_reset(Boustrophedon* env) {
    env->x = 0;
    env->y = 0;
    env->dx = 1;
    env->dy = 0;
    memset(env->coverage, 0, sizeof env->coverage);
    env->coverage[0][0] = 1;
    update_obs(env);
}

void c_step(Boustrophedon* env) {
    if (env->actions[0] == 1) {
        if (env->dx == 0 && env->dy == 1) {
            env->dx = 1;
            env->dy = 0;
        } else if (env->dx == 0 && env->dy == -1) {
            env->dx = -1;
            env->dy = 0;
        } else if (env->dx == 1 && env->dy == 0) {
            env->dx = 0;
            env->dy = -1;
        } else if (env->dx == -1 && env->dy == 0) {
            env->dx = 0;
            env->dy = 1;
        }
    }
    if (env->actions[0] == 2) {
        if (env->dx == 0 && env->dy == 1) {
            env->dx = -1;
            env->dy = 0;
        } else if (env->dx == 0 && env->dy == -1) {
            env->dx = 1;
            env->dy = 0;
        } else if (env->dx == 1 && env->dy == 0) {
            env->dx = 0;
            env->dy = 1;
        } else if (env->dx == -1 && env->dy == 0) {
            env->dx = 0;
            env->dy = -1;
        }
    }
    env->x += env->dx;
    env->y += env->dy;

    bool failure = false;
    bool success = true;
    float reward = 0;

    if (env->x < 0 || env->x >= SIZE || env->y < 0 || env->y >= SIZE) {
        failure = true;
        success = false;
    } else {
        if (!env->coverage[env->y][env->x]) {
            env->coverage[env->y][env->x] = true;
            reward += 0.1;
        }
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                if (!env->coverage[r][c]) {
                    success = false;
                }
            }
        }
    }

    if (success || failure) {
        env->terminals[0] = 1;
        env->log.n++;
        if (success) {
            env->log.score++;
            reward += 1.0f;
        } else {
            reward -= 1.0f;
        }
        c_reset(env);
    } else {
        env->terminals[0] = 0;
        update_obs(env);
    }
    env->rewards[0] = reward;
}

void c_render(Boustrophedon* env) {
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_WINDOW_HIGHDPI);
        InitWindow(SIZE * 10, SIZE * 10, "PufferLib Template");
        SetTargetFPS(5);
    }

    if (WindowShouldClose()) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (env->coverage[r][c]) {
                DrawRectangle(c * 10, r * 10, 10, 10, PUFF_CYAN);
            }
        }
    }
    DrawRectangle(env->x * 10, env->y * 10, 10, 10, PUFF_WHITE);

    EndDrawing();
}

void c_close(Boustrophedon* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
