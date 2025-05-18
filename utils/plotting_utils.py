import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_average_episode_return_across_seeds(log_paths, NUM_SEEDS, window=50, agent_names=['aif_reward', 'ql_reward'],k=20):

    dfs = []
    for seed in range(NUM_SEEDS):
        df = pd.read_csv(os.path.join(log_paths["infos"], f"log_seed_{seed}.csv"))
        df['seed'] = seed
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)

    # 1️⃣ Compute total return per episode per seed (sum rewards across steps)
    episode_returns = (
        all_data.groupby(['seed', 'episode'])[[agent_names[0], agent_names[1]]]
        .sum()
        .reset_index()
    )

    # 2️⃣ Compute mean and SEM across seeds
    mean_returns = episode_returns.groupby('episode')[[agent_names[0], agent_names[1]]].mean()
    sem_returns = episode_returns.groupby('episode')[[agent_names[0], agent_names[1]]].sem()

    # 3️⃣ Compute 95% CI
    ci95 = 1.96 * sem_returns

    # 4️⃣ Optional smoothing
    mean_returns[f"{agent_names[0]}_smooth"] = mean_returns[agent_names[0]].rolling(window, min_periods=1).mean()
    mean_returns[f"{agent_names[1]}_smooth"] = mean_returns[agent_names[1]].rolling(window, min_periods=1).mean()

    ci95[f"{agent_names[0]}_smooth"] = ci95[agent_names[0]].rolling(window, min_periods=1).mean()
    ci95[f"{agent_names[1]}_smooth"] = ci95[agent_names[1]].rolling(window, min_periods=1).mean()

    # 5️⃣ Plot
    plt.figure(figsize=(12, 6))

    plt.plot(mean_returns.index, mean_returns[f"{agent_names[0]}_smooth"], label=f"{agent_names[0]} Return", color='green')
    plt.fill_between(
        mean_returns.index,
        np.float64(mean_returns[f"{agent_names[0]}_smooth"] - ci95[f"{agent_names[0]}_smooth"]),
        np.float64(mean_returns[f"{agent_names[0]}_smooth"] + ci95[f"{agent_names[0]}_smooth"]),
        color='green', alpha=0.2
    )

    plt.plot(mean_returns.index, mean_returns[f"{agent_names[1]}_smooth"], label=f"{agent_names[1]} Return", color='orange')
    plt.fill_between(
        mean_returns.index,
        np.float64(mean_returns[f"{agent_names[1]}_smooth"] - ci95[f"{agent_names[1]}_smooth"]),
        np.float64(mean_returns[f"{agent_names[1]}_smooth"] + ci95[f"{agent_names[1]}_smooth"]),
        color='orange', alpha=0.2
    )

    plt.xlabel('Episode')
    plt.ylabel('Average Return (per episode)')
    plt.title('Average Returns per Episode with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)

    # Vertical lines every 50 episodes to mark environment switches
    for ep in range(0, mean_returns.index.max() + 1, k):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(log_paths["plots"], "average_returns_per_episode.png"))
    plt.show()




def plot_episode_return_for_one_seed_csv(csv_path, agent_cols=["aif_reward", "rand_reward"], save_path=None, smooth_window=50):
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
    
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    
    
    


# def plot_step_return_for_one_seed_csv(csv_path, agent_cols=["aif_reward", "rand_reward"], save_path=None, smooth_window=50):
#     """
#     Plot return per step for one seed file, with vertical lines separating episodes.
    
#     Args:
#         csv_path (str): Path to the CSV file.
#         agent_cols (list): List of columns containing reward per step (e.g., ["ql_reward", "random_reward"]).
#         save_path (str or None): If given, saves the plot to this path.
#     """
#     # Load CSV
#     df = pd.read_csv(csv_path)

#     # Compute return per step
#     df['step_return'] = df[agent_cols].cumsum(axis=1)

#     # Apply smoothing
#     # smoothed_returns = df['step_return'].rolling(window=smooth_window, min_periods=1).mean()

#     # Plot
#     plt.figure(figsize=(12, 6))
#     plt.plot(df.index, df['step_return'], label="Smoothed Step Return", color='b')

#     # Add vertical lines to separate episodes
#     episode_starts = df[df['episode'] != df['episode'].shift(1)].index
#     for episode_start in episode_starts:
#         plt.axvline(x=episode_start, color='r', linestyle='--', label="Episode Boundary" if episode_start == episode_starts[0] else "")

#     plt.xlabel("Step")
#     plt.ylabel(" Return")
#     plt.title("Step Return per Agent with Episode Boundaries")
#     plt.legend(loc='upper right')
#     plt.grid(True)

#     # Save or show the plot
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Plot saved to: {save_path}")
#     else:
#         plt.show()







def plot_step_return_for_one_seed_csv(csv_path, agent_cols=["aif_reward", "rand_reward"], save_path=None):
    """
    Plot cumulative reward (return) per step for one seed file, with vertical lines separating episodes.
    
    Args:
        csv_path (str): Path to the CSV file.
        agent_cols (list): List of columns containing reward per step (e.g., ["ql_reward", "random_reward"]).
        save_path (str or None): If given, saves the plot to this path.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Compute reward per step (sum of rewards for each agent column)
    df['step_reward'] = df[agent_cols].sum(axis=1)

    # Compute cumulative reward (return) per step
    df['cumulative_reward'] = df['step_reward'].cumsum()

    # Plot the cumulative reward for each step
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_reward'], label="Cumulative Reward", color='b')

    # Add vertical lines to separate episodes
    episode_starts = df[df['episode'] != df['episode'].shift(1)].index
    for episode_start in episode_starts:
        plt.axvline(x=episode_start, color='r', linestyle='--', label="Episode Boundary" if episode_start == episode_starts[0] else "")

    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward per Step with Episode Boundaries")
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
