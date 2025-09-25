# run_demo.py
"""
Final version of the demonstration script.

This script finds all saved model checkpoints (e.g., dqn_cart_ep200.pth)
in the 'models' directory and generates a separate demo video for each one.
It uses the `is_demo=True` flag in the environment to ensure the videos
are full-length and properly showcase the agent's learned behavior.
"""

import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import re

# Import the necessary classes from your other project scripts
from cart_env import SupermarketEnv
from train_dqn import DQNAgent, Config

# --- Configuration ---
MODEL_DIR = Config.MODEL_DIR
VIDEO_FOLDER = Path("videos")
VIDEO_FOLDER.mkdir(exist_ok=True) # Ensure the video directory exists

def run_multi_demos():
    """
    Loads all saved agent checkpoints from the 'models' directory and records
    a separate demonstration video for each one.
    """
    
    # 1. Find all saved model checkpoints and sort them chronologically
    model_files = sorted(
        list(MODEL_DIR.glob("dqn_cart_ep*.pth")) +
        list(MODEL_DIR.glob("dqn_cart_final.pth"))
    )

    if not model_files:
        print(f"Error: No model files found in {MODEL_DIR}")
        print("Please run the training script (train_dqn.py) first to generate the models.")
        return

    # To avoid creating an environment inside the loop, get dimensions once
    try:
        dummy_env = SupermarketEnv()
        obs_dim = dummy_env.observation_space.shape[0]
        action_dim = dummy_env.action_space.n
        dummy_env.close()
    except Exception as e:
        print(f"Failed to create a dummy environment to get dimensions: {e}")
        return

    print(f"Found {len(model_files)} model checkpoints. Generating demos...")

    # 2. Loop through each found model file
    for model_path in model_files:
        # Extract episode number from the filename for clear video naming
        match = re.search(r"ep(\d+)", model_path.name)
        episode_num = match.group(1) if match else "final"
        video_prefix = f"cart-rl-demo-ep{episode_num}"
        
        print(f"\n--- Running demo for model: {model_path.name} ---")

        # 3. Set up the environment with demo mode for a full-length episode
        env = SupermarketEnv(render_mode="rgb_array", is_demo=True)
        env = RecordVideo(
            env,
            video_folder=str(VIDEO_FOLDER),
            name_prefix=video_prefix,
            episode_trigger=lambda x: True  # Record this single episode
        )
        
        # 4. Initialize the agent and load the weights for this checkpoint
        device = torch.device("cpu")
        agent = DQNAgent(obs_dim, action_dim, device)
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        agent.policy_net.eval() # Set the network to evaluation mode
        
        # 5. Run one full episode
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Use eps=0.0 for deterministic behavior (no random actions)
            action = agent.select_action(obs, eps=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Demo finished for Episode {episode_num}. Total reward: {total_reward:.2f}")
        # env.close() is crucial to finalize and save the video file for this loop
        env.close()

    print(f"\nAll demos generated! Check the '{VIDEO_FOLDER}' directory.")

if __name__ == "__main__":
    run_multi_demos()

