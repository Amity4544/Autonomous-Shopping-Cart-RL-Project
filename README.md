Of course. My apologies for the previous issues. Here is the full, complete, and final version of the `README.md` file directly in the chat, accurately reflecting all the features of your project.

-----

# Autonomous Shopping Cart Follower using Deep Reinforcement Learning

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)

A Deep Q-Network (DQN) agent trained from scratch to autonomously control a shopping cart, enabling it to follow a human target through a simulated supermarket environment while avoiding obstacles.

-----

## Table of Contents

  - [About The Project]
  - [Architecture]
  - [Getting Started]
      - [Prerequisites]
      - [Installation]
  - [Usage]
      - [1. Train the Agent]
      - [2. Plot the Training Results]
      - [3. Run the Demo Videos]
  - [RL Formulation]
      - [State Space]
      - [Action Space]
      - [Reward Function]
  - [Code Structure]
  - [Limitations & Future Work]
  - [License]
-----

## About The Project

This project implements an autonomous shopping cart agent using Deep Reinforcement Learning. The primary goal is to train an agent that can intelligently follow a person through a simulated supermarket, navigating aisles and avoiding shelves without explicit instructions.

The entire system is built from the ground up, including:

  - A custom 2D supermarket environment built with **Pygame**.
  - A Deep Q-Network (DQN) agent implemented with **PyTorch**.
  - A complete pipeline for training, evaluation, and visualization.

The agent learns its behavior through trial and error, guided by a carefully designed reward function. Over thousands of training episodes, it develops a sophisticated policy to balance following its target, maintaining a safe distance, and avoiding collisions.

-----

## Architecture

The project follows a standard Reinforcement Learning feedback loop.

1.  **Observation**: The **Environment** provides the current `State` to the **Agent**.
2.  **Action**: The **Agent's Policy Network** processes the state and selects an `Action`.
3.  **Feedback**: The **Environment** executes the action and returns a `Reward` and the `Next State`.
4.  **Learning**: The agent stores this experience `(state, action, reward, next_state)` in its **Replay Buffer**. The **DQN algorithm** samples from this buffer to update its policy network, improving its decision-making over time.

-----

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

  - Python (version 3.9 or higher)
  - `pip` and `venv` for package management

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/shopping-cart-rl.git
    cd shopping-cart-rl
    ```

2.  **Create and activate a virtual environment:**

    ```sh
    # Create the virtual environment
    python -m venv venv

    # Activate it (Windows)
    .\venv\Scripts\activate

    # Activate it (macOS/Linux)
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file should be created with the following content:

    ```txt
    torch
    gymnasium
    pygame
    numpy
    pandas
    matplotlib
    moviepy
    ```

    Then, run the installation command:

    ```sh
    pip install -r requirements.txt
    ```

-----

## Usage

The project workflow is divided into three main steps: training, plotting, and running the demo.

### 1\. Train the Agent

Execute the training script from the root directory. This will start the training process using the settings defined in the `Config` class.

```sh
python src/train_dqn.py
```

  - **Output:**
      - The trained model checkpoints will be saved periodically in the `/models` directory (e.g., `dqn_cart_ep200.pth`).
      - The final trained model will be saved as `models/dqn_cart_final.pth`.
      - All performance metrics (rewards, episode length, collisions) will be logged to `training_metrics.csv`.

### 2\. Plot the Training Results

After training is complete, run the plotting script to visualize the agent's performance.

```sh
python src/plot_results.py
```

  - **Output:**
      - A 2x2 grid of plots will be displayed on the screen, showing the learning curves for all tracked metrics.
      - The plot image will be saved to `plots/training_performance_full.png`.

### 3\. Run the Demo Videos

Execute the demo script to generate videos of your trained agent at different stages of learning.

```sh
python src/run_demo.py
```

  - **Output:**
      - The script will find all model checkpoints in the `/models` directory.
      - It will generate a separate `.mp4` video for each checkpoint and save them in the `/videos` directory.

-----

## RL Formulation

The problem is modeled as a Markov Decision Process (MDP) with the following components:

### State Space

The state is a 4-element vector of normalized floating-point numbers, giving the agent a complete picture of its situation.
`s = [relative_x, relative_y, cart_x_norm, cart_y_norm]`

  - **`relative_x, relative_y`**: The normalized distance and direction from the cart to the human target.
  - **`cart_x_norm, cart_y_norm`**: The cart's own normalized coordinates on the map, providing self-awareness.

### Action Space

The agent can choose from 5 discrete actions at each timestep:

  - `0`: Stop
  - `1`: Move Up
  - `2`: Move Down
  - `3`: Move Left
  - `4`: Move Right

### Reward Function

The reward function is carefully shaped to guide the agent towards the desired behavior.

| Component                 | Value              | Purpose                                                 |
| ------------------------- | ------------------ | ------------------------------------------------------- |
| `IDEAL_ZONE_BONUS`        | **`+5.0`** | Encourage staying in the optimal following distance.      |
| `REDUCE_DISTANCE_BONUS`   | **`+0.5`** | Provide frequent positive feedback for getting closer.  |
| `DISTANCE_PENALTY`        | **`-0.1 * dist`** | Penalize being too far from the target.                 |
| `COLLISION_PENALTY`       | **`-10.0`** | Strongly discourage crashing into shelves.              |
| `HUMAN_COLLISION_PENALTY` | **`-20.0`** | A large penalty for hitting the human; ends the episode.|
| `TIME_PENALTY`            | **`-0.01`** | A small penalty per step to encourage efficiency.       |

-----

## Code Structure

The project is organized into four main Python scripts within the `src/` directory.

  - **`src/cart_env.py`**:

      - Defines the `SupermarketEnv` class, which manages the simulation.
      - Handles game logic, state transitions, and reward calculation.
      - Includes a special `is_demo` mode for generating full-length videos.
      - Tracks and reports episode length and collision metrics.

  - **`src/train_dqn.py`**:

      - Contains the core training logic.
      - Defines the deeper `QNetwork` and the `DQNAgent` class, which encapsulates the DQN algorithm.
      - Manages the training loop, saving models and logging all performance metrics to a CSV file.

  - **`src/plot_results.py`**:

      - A utility script for visualization.
      - Reads `training_metrics.csv` using Pandas.
      - Generates and saves a 2x2 grid of performance plots using Matplotlib.

  - **`src/run_demo.py`**:

      - The script for evaluating the final agent.
      - Finds all saved model checkpoints in the `/models` directory.
      - Runs a full-length episode for each checkpoint and saves the output as an `.mp4` video.

-----

## Limitations & Future Work

While this project was successful, there are several exciting avenues for future improvements:

  - **Enhance Human Behavior Simulation**: A more realistic simulation would involve the human occasionally stopping for several seconds in front of a shelf, mimicking real shopping behavior. This would challenge the agent to learn to wait patiently.
  - **Implement Dueling DQN**: Upgrade the model architecture to a Dueling DQN. This is a relatively simple change that can often lead to faster learning and better final performance by separating the estimation of state values and action advantages.
  - **Add a "Safety Bubble" Reward**: Introduce a small negative reward for getting too close to a shelf, even without a collision. This would encourage the agent to proactively stay in the center of aisles, making its navigation safer and smoother.

-----
"# Autonomous-Shopping-Cart-RL-Project" 
