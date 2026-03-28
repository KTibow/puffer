import math

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.roomba import binding

WIDTH = 1000.0  # mm
HEIGHT = 1000.0  # mm
SPEED = 250  # mm/s
DT = 0.1  # s
ROBOT_DIAMETER = 329.9  # mm
ROBOT_RADIUS = ROBOT_DIAMETER / 2.0
COVERAGE_RADIUS = 90.0  # Keep in sync with roomba.h
GRID_COLS = 12  # Keep in sync with roomba.h
GRID_ROWS = 12  # Keep in sync with roomba.h
N_DOTS = GRID_COLS * GRID_ROWS
N_NEAREST_DOTS = 2
OBS_COVERAGE_OFFSET = 11
SUCCESS_COVERAGE = 0.90
EPISODE_SECONDS = 80
TICK_LIMIT = int(EPISODE_SECONDS / DT)
WHEEL_BASE = 235

def _decode_pose(obs):
    x = float(obs[0]) * WIDTH
    y = float(obs[1]) * HEIGHT
    cos_heading = float(obs[2]) * 2.0 - 1.0
    sin_heading = float(obs[3]) * 2.0 - 1.0
    bearing = math.atan2(sin_heading, cos_heading)
    return x, y, bearing


def _decode_nearest_pointer(obs, slot=0):
    distance = float(obs[5 + slot * 3]) * math.hypot(WIDTH, HEIGHT)
    cos_rel = float(obs[6 + slot * 3]) * 2.0 - 1.0
    sin_rel = float(obs[7 + slot * 3]) * 2.0 - 1.0
    return distance, cos_rel, sin_rel


def _wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _wheel_controller(distance, angle_error):
    if distance < 1e-6:
        return np.zeros((1, 2), dtype=np.float32)

    turn = np.clip(angle_error / (math.pi / 3.0), -1.0, 1.0)
    if abs(angle_error) > math.pi / 5.0:
        forward = 0.0
    else:
        forward = np.clip(
            (distance - COVERAGE_RADIUS * 0.35) / (SPEED * DT * 3.0), 0.0, 1.0
        )
        forward *= max(0.0, 1.0 - abs(angle_error) / (math.pi / 2.0))

    left = np.clip(forward + turn, -1.0, 1.0)
    right = np.clip(forward - turn, -1.0, 1.0)
    return np.array([[left, right]], dtype=np.float32)


def _steer_to_target(x, y, bearing, target_x, target_y):
    dx = target_x - x
    dy = target_y - y
    distance = math.hypot(dx, dy)
    angle_error = _wrap_angle(math.atan2(dy, dx) - bearing)
    return _wheel_controller(distance, angle_error)


class NearestDotBaseline:
    def reset(self):
        return None

    def act(self, observations):
        obs = observations[0]
        _x, _y, _bearing = _decode_pose(obs)
        distance, cos_rel, sin_rel = _decode_nearest_pointer(obs, slot=0)
        if distance <= 1e-6:
            return np.zeros((1, 2), dtype=np.float32)

        angle_error = math.atan2(sin_rel, cos_rel)
        return _wheel_controller(distance, angle_error)


class BoustrophedonBaseline:
    def __init__(self, lane_spacing=None):
        self.margin = COVERAGE_RADIUS
        self.lane_spacing = lane_spacing or (COVERAGE_RADIUS * 1.75)
        self.cleanup = NearestDotBaseline()
        self.waypoints = self._make_waypoints()
        self.reset()

    def _make_waypoints(self):
        ys = np.arange(self.margin, HEIGHT - self.margin + 1e-6, self.lane_spacing)
        waypoints = []
        for lane_idx, y in enumerate(ys):
            if lane_idx % 2 == 0:
                x0, x1 = self.margin, WIDTH - self.margin
            else:
                x0, x1 = WIDTH - self.margin, self.margin

            if lane_idx == 0:
                waypoints.append((x1, y))
            else:
                waypoints.append((x0, y))
                waypoints.append((x1, y))
        return waypoints

    def reset(self):
        self.waypoint_idx = 0
        self.cleanup.reset()

    def act(self, observations):
        obs = observations[0]
        x, y, bearing = _decode_pose(obs)

        while self.waypoint_idx < len(self.waypoints):
            target_x, target_y = self.waypoints[self.waypoint_idx]
            if math.hypot(target_x - x, target_y - y) <= COVERAGE_RADIUS * 0.4:
                self.waypoint_idx += 1
            else:
                break

        if self.waypoint_idx >= len(self.waypoints):
            return self.cleanup.act(observations)

        remaining_fraction = 1.0 - float(obs[4])
        if remaining_fraction < 0.20:
            return self.cleanup.act(observations)

        target_x, target_y = self.waypoints[self.waypoint_idx]
        return _steer_to_target(x, y, bearing, target_x, target_y)


def evaluate_baseline(agent, episodes=10, seed=0, diverse_resets=False):
    env = Roomba(num_envs=1, seed=seed, diverse_resets=diverse_resets)
    observations, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()

    coverages = []
    successes = []
    while len(coverages) < episodes:
        actions = agent.act(observations)
        observations, rewards, terminals, truncations, info = env.step(actions)
        if terminals[0]:
            if info[0]:
                coverages.append(float(info[0].get("coverage", 0.0)))
                successes.append(float(info[0].get("perf", 0.0)))
            if hasattr(agent, "reset"):
                agent.reset()

    env.close()
    return {
        "coverage_mean": float(np.mean(coverages)),
        "coverage_std": float(np.std(coverages)),
        "success_rate": float(np.mean(successes)),
        "episodes": episodes,
    }


BASELINES = {
    "nearest_dot": NearestDotBaseline,
    "boustrophedon": BoustrophedonBaseline,
}


class Roomba(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        log_interval=128,
        buf=None,
        seed=0,
        diverse_resets=True,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=((OBS_COVERAGE_OFFSET + N_DOTS),),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            width=WIDTH,
            height=HEIGHT,
            speed=SPEED,
            dt=DT,
            tick_limit=TICK_LIMIT,
            wheel_base=WHEEL_BASE,
            diverse_resets=int(diverse_resets),
        )

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)
