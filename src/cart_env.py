# cart_env.py
"""
Final version of the Supermarket Cart Environment.

This script defines the Gymnasium-compliant environment for the RL agent.
It includes a special 'is_demo' flag in its constructor. When set to True,
the episode does not terminate upon human collision, allowing for the
recording of full-length demonstration videos. It also tracks and reports
key metrics like episode length and collisions in the 'info' dictionary
upon an episode's conclusion.
"""

import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional

# --- Configuration Constants ---
GRID_COLS, GRID_ROWS = 60, 25
CELL_SIZE = 20
SCREEN_W, SCREEN_H = GRID_COLS * CELL_SIZE, GRID_ROWS * CELL_SIZE
BG_COLOR = (240, 240, 240)
SHELF_COLOR = (100, 100, 100)
CART_COLOR = (0, 200, 0)
TARGET_COLOR = (200, 0, 0)
IDEAL_DISTANCE_MIN = 1.5
IDEAL_DISTANCE_MAX = 3.0
MIN_FOLLOW_DISTANCE = 2.0
DISTANCE_PENALTY_SCALE = -0.1
REDUCE_DISTANCE_BONUS = 0.5
IDEAL_ZONE_BONUS = 5.0
TIME_PENALTY = -0.01
COLLISION_PENALTY = -10.0
HUMAN_COLLISION_PENALTY = -20.0
MAX_EPISODE_STEPS = 500
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class SupermarketEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str = "rgb_array", is_demo: bool = False):
        super().__init__()
        self.render_mode = render_mode
        self.is_demo = is_demo
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.grid = self._create_grid()
        self.cart_pos: Optional[np.ndarray] = None
        self.target_pos: Optional[np.ndarray] = None
        self.target_dir: Optional[Tuple[int, int]] = None
        self.steps_in_dir = 0
        self.current_step = 0
        self.collisions = 0
        self.screen = None
        self.clock = None
        self.font = None

    def _create_grid(self) -> List[List[str]]:
        grid = [['.' for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        for col_start in [8, 20, 32, 44]:
            for row in range(5, GRID_ROWS - 8):
                for c in range(col_start, col_start + 4): grid[row][c] = '#'
        for row in range(0, 3): grid[row] = ['#'] * GRID_COLS
        for row in range(GRID_ROWS - 4, GRID_ROWS - 2):
            for col in range(10, GRID_COLS - 10): grid[row][col] = '#'
        return grid

    def _is_valid_pos(self, pos: np.ndarray) -> bool:
        x, y = pos
        return 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS and self.grid[y][x] != '#'

    def _get_random_valid_pos(self) -> np.ndarray:
        while True:
            pos = self.np_random.integers([1, 3], [GRID_COLS - 1, GRID_ROWS - 3])
            if self._is_valid_pos(pos): return pos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.target_pos = self._get_random_valid_pos()
        cart_placed = False
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * np.pi)
            distance = self.np_random.uniform(MIN_FOLLOW_DISTANCE, IDEAL_DISTANCE_MAX + 1.5)
            offset = np.array([distance * np.cos(angle), distance * np.sin(angle)]).round().astype(int)
            potential_cart_pos = self.target_pos + offset
            if self._is_valid_pos(potential_cart_pos):
                self.cart_pos = potential_cart_pos
                cart_placed = True
                break
        if not cart_placed: self.cart_pos = self._get_random_valid_pos()
        self.target_dir = random.choice(DIRECTIONS)
        self.steps_in_dir, self.current_step, self.collisions = 0, 0, 0
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        relative_pos = (self.target_pos - self.cart_pos) / np.array([GRID_COLS, GRID_ROWS])
        cart_pos_norm = (self.cart_pos / np.array([GRID_COLS, GRID_ROWS])) * 2 - 1
        return np.concatenate([relative_pos, cart_pos_norm]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        info = {"distance": np.linalg.norm(self.target_pos - self.cart_pos)}
        terminated = np.array_equal(self.cart_pos, self.target_pos)
        truncated = self.current_step >= MAX_EPISODE_STEPS
        if (terminated or truncated):
            info["episode_length"] = self.current_step
            info["collisions"] = self.collisions
        return info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        prev_dist = np.linalg.norm(self.target_pos - self.cart_pos)
        self._move_human()
        collision, human_collision = self._move_cart(action)
        current_dist = np.linalg.norm(self.target_pos - self.cart_pos)
        reward = self._calculate_reward(current_dist, prev_dist, collision, human_collision)
        
        terminated = human_collision
        if self.is_demo:
            terminated = False

        truncated = self.current_step >= MAX_EPISODE_STEPS
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _move_human(self):
        self.steps_in_dir += 1
        if self.steps_in_dir > self.np_random.integers(6, 20) or self.np_random.random() < 0.1:
            self.target_dir = random.choice(DIRECTIONS)
            self.steps_in_dir = 0
        next_pos = self.target_pos + np.array(self.target_dir)
        if self._is_valid_pos(next_pos): self.target_pos = next_pos
        else: self.target_dir = random.choice(DIRECTIONS); self.steps_in_dir = 0

    def _move_cart(self, action: int) -> Tuple[bool, bool]:
        action_to_move = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        move = np.array(action_to_move[action])
        next_pos = self.cart_pos + move
        shelf_collision = not self._is_valid_pos(next_pos)
        human_collision = np.array_equal(next_pos, self.target_pos)
        if shelf_collision:
            self.collisions += 1
        else:
            self.cart_pos = next_pos
        return shelf_collision, human_collision

    def _calculate_reward(self, dist: float, prev_dist: float, collision: bool, human_collision: bool) -> float:
        reward = 0.0
        reward += dist * DISTANCE_PENALTY_SCALE
        if dist < prev_dist: reward += REDUCE_DISTANCE_BONUS
        if IDEAL_DISTANCE_MIN < dist <= IDEAL_DISTANCE_MAX: reward += IDEAL_ZONE_BONUS
        if collision: reward += COLLISION_PENALTY
        if human_collision: reward += HUMAN_COLLISION_PENALTY
        reward += TIME_PENALTY
        return reward

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                pygame.display.set_caption("Supermarket Cart RL")
            else: self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.close()
        self.screen.fill(BG_COLOR)
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == '#': pygame.draw.rect(self.screen, SHELF_COLOR, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, CART_COLOR, (self.cart_pos[0] * CELL_SIZE, self.cart_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.circle(self.screen, TARGET_COLOR, (self.target_pos[0] * CELL_SIZE + CELL_SIZE / 2, self.target_pos[1] * CELL_SIZE + CELL_SIZE / 2), CELL_SIZE / 2 - 2)
        dist = np.linalg.norm(self.target_pos - self.cart_pos)
        text = self.font.render(f"Step: {self.current_step} | Distance: {dist:.2f}", True, (10, 10, 10))
        self.screen.blit(text, (10, 10))
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else: return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

