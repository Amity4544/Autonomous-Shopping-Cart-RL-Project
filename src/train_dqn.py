# train_dqn.py
"""
Final version of the training script for the DQN agent.

This script orchestrates the training process. It uses the more robust,
deeper QNetwork architecture and saves all key performance metrics to a
CSV file for later analysis and plotting. It saves model checkpoints
every 200 episodes.
"""

import csv
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from pathlib import Path
from gymnasium.vector import SyncVectorEnv
from typing import List

from cart_env import SupermarketEnv

class Config:
    NUM_ENVS: int = 4
    MAX_EPISODES: int = 1000
    BATCH_SIZE: int = 256
    GAMMA: float = 0.99
    LEARNING_RATE: float = 1e-4
    BUFFER_SIZE: int = 100_000
    TARGET_UPDATE_INTERVAL: int = 1000
    EPS_START: float = 1.0
    EPS_END: float = 0.02
    EPS_DECAY_STEPS: int = 150_000
    MODEL_DIR: Path = Path("models")
    CHECKPOINT_FREQ: int = 200
    METRICS_CSV_PATH: Path = Path("training_metrics.csv")

Config.MODEL_DIR.mkdir(exist_ok=True)
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class QNetwork(nn.Module):
    """The neural network for Q-value approximation (Deeper Architecture)."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int): self.buffer = deque(maxlen=capacity)
    def add(self, exp: Experience): self.buffer.append(exp)
    def sample(self, bs: int) -> List[Experience]: return random.sample(self.buffer, bs)
    def __len__(self) -> int: return len(self.buffer)

class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device):
        self.action_dim = action_dim; self.device = device
        self.policy_net = QNetwork(obs_dim, action_dim).to(device)
        self.target_net = QNetwork(obs_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.LEARNING_RATE, amsgrad=True)
        self.buffer = ReplayBuffer(Config.BUFFER_SIZE); self.criterion = nn.SmoothL1Loss()
    def select_action(self, state: np.ndarray, eps: float) -> int:
        if random.random() < eps: return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_vals = self.policy_net(state_t)
                return q_vals.argmax().item()
    def learn(self) -> float or None:
        if len(self.buffer) < Config.BATCH_SIZE: return None
        exps = self.buffer.sample(Config.BATCH_SIZE); batch = Experience(*zip(*exps))
        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)
        curr_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * Config.GAMMA * next_q
        loss = self.criterion(curr_q, target_q)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()
    def update_target_net(self): self.target_net.load_state_dict(self.policy_net.state_dict())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    env_fns = [lambda: SupermarketEnv() for _ in range(Config.NUM_ENVS)]
    envs = SyncVectorEnv(env_fns)
    agent = DQNAgent(envs.observation_space.shape[1], envs.action_space.nvec[0], device)
    
    global_step, completed_episodes, start_time = 0, 0, time.time()
    episode_rewards = np.zeros(Config.NUM_ENVS)
    reward_history = deque(maxlen=100)
    
    csv_header = ["episode", "total_reward", "avg_reward_100", "epsilon", "episode_length", "collisions"]
    with open(Config.METRICS_CSV_PATH, "w", newline="") as f:
        csv_writer = csv.writer(f); csv_writer.writerow(csv_header)
    print(f"Metrics will be saved to {Config.METRICS_CSV_PATH}")

    states, _ = envs.reset()
    while completed_episodes < Config.MAX_EPISODES:
        eps = np.interp(global_step, [0, Config.EPS_DECAY_STEPS], [Config.EPS_START, Config.EPS_END])
        actions = [agent.select_action(states[i], eps) for i in range(Config.NUM_ENVS)]
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        dones = terminated | truncated
        for i in range(Config.NUM_ENVS): agent.buffer.add(Experience(states[i], actions[i], rewards[i], next_states[i], dones[i]))
        states = next_states; episode_rewards += rewards; global_step += Config.NUM_ENVS
        agent.learn()
        if global_step % Config.TARGET_UPDATE_INTERVAL == 0: agent.update_target_net()

        if "_final_info" in infos:
            for i, done in enumerate(infos["_final_info"]):
                if done:
                    completed_episodes += 1
                    finished_reward = episode_rewards[i]
                    reward_history.append(finished_reward)
                    avg_reward = np.mean(reward_history)
                    
                    final_info = infos["final_info"][i]
                    ep_len = final_info.get("episode_length", 0)
                    ep_collisions = final_info.get("collisions", 0)
                    
                    print(f"Ep: {completed_episodes}, Reward: {finished_reward:.2f}, Avg Reward: {avg_reward:.2f}, Len: {ep_len}, Collisions: {ep_collisions}")
                    
                    with open(Config.METRICS_CSV_PATH, "a", newline="") as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([completed_episodes, finished_reward, avg_reward, eps, ep_len, ep_collisions])

                    episode_rewards[i] = 0
                    if completed_episodes % Config.CHECKPOINT_FREQ == 0:
                        path = Config.MODEL_DIR / f"dqn_cart_ep{completed_episodes}.pth"
                        torch.save(agent.policy_net.state_dict(), path)
                        print(f"Checkpoint saved to {path}")
    
    final_path = Config.MODEL_DIR / "dqn_cart_final.pth"
    torch.save(agent.policy_net.state_dict(), final_path)
    print(f"\nTraining finished. Final model saved to {final_path}")
    envs.close()

if __name__ == "__main__":
    main()

