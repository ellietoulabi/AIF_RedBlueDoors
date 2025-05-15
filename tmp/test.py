import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_episode_return_from_seed_csv(csv_path, agent_cols=["aif_reward", "rand_reward"], save_path=None, smooth_window=50):
    """
    Plot return per episode for one seed file.
    
    Args:
        csv_path (str): Path to the CSV file.
        agent_cols (list): List of columns containing reward per step (e.g., ["ql_reward", "random_reward"]).
        save_path (str or None): If given, saves the plot to this path.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Compute return per episode
    episode_returns = df.groupby("episode")[agent_cols].sum()

    # Apply smoothing
    smoothed_returns = episode_returns.rolling(window=smooth_window, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(12, 6))
    for col in agent_cols:
        plt.plot(
            episode_returns.index,
            smoothed_returns[col],
            label=f"{col} (smoothed)",
        )

    plt.xlabel("Episode")
    plt.ylabel("Smoothed Return")
    plt.title("Episode Return per Agent (Smoothed)")
    plt.legend()
    plt.grid(True)
    if save_path:
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    csv_file = "/Users/el/Desktop/log_seed_0.csv"  # path to your file
    plot_episode_return_from_seed_csv(csv_file, save_path="return_per_episode_seed0.png",smooth_window=100)
