# plot_results.py
"""
Final version of the plotting script.

This script reads the 'training_metrics.csv' file and generates a 2x2
grid of performance plots using Matplotlib. The plots visualize the agent's
learning progress by showing average reward, episode length, collisions,
and epsilon decay over the course of training.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
METRICS_CSV_PATH = Path("training_metrics.csv")
PLOTS_DIR = Path("plots")
OUTPUT_PLOT_FILE = PLOTS_DIR / "training_performance_full.png"

# Ensure the directory for saving plots exists
PLOTS_DIR.mkdir(exist_ok=True)

def plot_metrics():
    """Reads the metrics CSV and generates a 2x2 grid of performance plots."""
    
    # 1. Check if the metrics file exists to provide a helpful error message
    if not METRICS_CSV_PATH.exists():
        print(f"Error: Metrics file not found at '{METRICS_CSV_PATH}'")
        print("Please run the training script (train_dqn.py) first to generate the data.")
        return

    # 2. Read the data using pandas and calculate rolling averages
    try:
        df = pd.read_csv(METRICS_CSV_PATH)
        # Calculate rolling averages to smooth the graphs for better trend visualization
        df['collisions_rolling_avg'] = df['collisions'].rolling(window=100, min_periods=1).mean()
        df['length_rolling_avg'] = df['episode_length'].rolling(window=100, min_periods=1).mean()
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return

    print(f"Successfully loaded and processed metrics from {METRICS_CSV_PATH}")

    # 3. Create the 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('DQN Agent Training Performance', fontsize=20)

    # Plot 1: Average Reward
    axs[0, 0].plot(df['episode'], df['avg_reward_100'], label='Avg Reward (100-ep roll)', color='blue')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].set_title('Average Episode Reward')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Episode Length
    axs[0, 1].plot(df['episode'], df['length_rolling_avg'], label='Avg Length (100-ep roll)', color='purple')
    axs[0, 1].set_ylabel('Steps')
    axs[0, 1].set_title('Average Episode Length')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Collisions
    axs[1, 0].plot(df['episode'], df['collisions_rolling_avg'], label='Avg Collisions (100-ep roll)', color='red')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Number of Collisions')
    axs[1, 0].set_title('Average Shelf Collisions')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 4: Epsilon Decay
    axs[1, 1].plot(df['episode'], df['epsilon'], label='Epsilon', color='green')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Epsilon (Exploration Rate)')
    axs[1, 1].set_title('Epsilon Decay')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    # 4. Adjust layout, save the figure, and show it
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PLOT_FILE)
    
    print(f"Plot saved successfully to '{OUTPUT_PLOT_FILE}'")
    
    plt.show()

if __name__ == "__main__":
    plot_metrics()

