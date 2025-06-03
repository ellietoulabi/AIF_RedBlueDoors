import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_average_episode_return_across_seeds(
    log_paths,
    NUM_SEEDS,
    window=50,
    agent_names=['aif_reward', 'rand_reward'],
    k=20
):
    """
    Plots the average episode return (±95% CI) across multiple seeds for the specified agents.
    
    Args:
        log_paths (dict): Dictionary with keys "infos" (path to CSV logs) and "plots" (path to save plots).
        NUM_SEEDS (int): Number of seed files to load and average over.
        window (int): Rolling window size for smoothing the return curves.
        agent_names (list of str): Column names for agents' reward values in the CSV.
        k (int): Interval (in episodes) at which to draw vertical dashed lines.
    """
    # Load and concatenate all seeds' data
    dfs = []
    for seed in range(NUM_SEEDS):
        df = pd.read_csv(os.path.join(log_paths["infos"], f"log_seed_{seed}.csv"))
        df['seed'] = seed
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)

    # Compute per-seed, per-episode returns (sum of rewards)
    episode_returns = (
        all_data.groupby(['seed', 'episode'])[[agent_names[0], agent_names[1]]]
        .sum()
        .reset_index()
    )

    # Compute across-seeds statistics: mean, SEM, 95% CI
    mean_returns = episode_returns.groupby('episode')[[agent_names[0], agent_names[1]]].mean()
    sem_returns = episode_returns.groupby('episode')[[agent_names[0], agent_names[1]]].sem()
    ci95_returns = 1.96 * sem_returns

    # Optional smoothing (rolling mean) for returns and CI
    for agent in agent_names:
        mean_returns[f"{agent}_smooth"] = mean_returns[agent].rolling(window, min_periods=1).mean()
        ci95_returns[f"{agent}_smooth"] = ci95_returns[agent].rolling(window, min_periods=1).mean()

    # Plotting average return with 95% CI
    episodes = mean_returns.index.values
    plt.figure(figsize=(12, 6))
    for idx, agent in enumerate(agent_names):
        sm = mean_returns[f"{agent}_smooth"]
        ci = ci95_returns[f"{agent}_smooth"]
        color = f"C{idx}"
        plt.plot(episodes, sm, label=f"{agent} Return", color=color)
        plt.fill_between(
            episodes,
            sm - ci,
            sm + ci,
            color=color,
            alpha=0.2
        )
    plt.xlabel('Episode')
    plt.ylabel('Average Return (per episode)')
    plt.title('Average Returns per Episode with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)

    # Vertical lines every k episodes to mark environment switches
    for ep in range(0, episodes.max() + 1, k):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(log_paths["plots"], "average_returns_per_episode.png")
    plt.savefig(save_path)
    plt.show()


def plot_success_rate_across_seeds(
    log_paths,
    NUM_SEEDS,
    window=50,
    agent_names=['aif_reward', 'rand_reward'],
    k=20
):
    """
    Plots the episode success rate (±95% CI) across multiple seeds for specified agents,
    where 'success' is defined as any step in an episode having a reward of 1.
    
    Args:
        log_paths (dict): Dictionary with keys "infos" (path to CSV logs) and "plots" (path to save plots).
        NUM_SEEDS (int): Number of seed files to load and average over.
        window (int): Rolling window size for smoothing the success rate curve.
        agent_names (list of str): Column names for agents' reward values in the CSV.
        k (int): Interval (in episodes) at which to draw vertical dashed lines.
    """
    # Load and concatenate all seeds' data
    dfs = []
    for seed in range(NUM_SEEDS):
        df = pd.read_csv(os.path.join(log_paths["infos"], f"log_seed_{seed}.csv"))
        df['seed'] = seed
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)

    # Compute per-seed, per-episode success: 1 if any step has reward == 1
    episode_success = all_data.groupby(['seed', 'episode'])[agent_names].max().reset_index()
    # Convert to binary success indicator (1 if max reward >= 1, else 0)
    for agent in agent_names:
        episode_success[f"{agent}_success"] = (episode_success[agent] >= 1).astype(int)

    # Compute across-seeds statistics: success rate, SEM, 95% CI
    success_rates = {
        agent: episode_success.groupby('episode')[f"{agent}_success"].mean().rename(f"{agent}_rate")
        for agent in agent_names
    }
    sem_success = {
        agent: episode_success.groupby('episode')[f"{agent}_success"].sem().rename(f"{agent}_sem")
        for agent in agent_names
    }
    ci95_success = {agent: 1.96 * sem_success[agent] for agent in agent_names}

    # Optional smoothing (rolling mean) for success rates and CI
    success_rate_smooth = {}
    ci95_success_smooth = {}
    for agent in agent_names:
        success_rate_smooth[agent] = success_rates[agent].rolling(window, min_periods=1).mean()
        ci95_success_smooth[agent] = ci95_success[agent].rolling(window, min_periods=1).mean()

    # Plotting success rate with 95% CI for each agent
    episodes = success_rates[agent_names[0]].index.values
    plt.figure(figsize=(12, 6))
    for idx, agent in enumerate(agent_names):
        sr_smooth = success_rate_smooth[agent]
        ci = ci95_success_smooth[agent]
        color = f"C{idx}"
        plt.plot(episodes, sr_smooth, label=f"{agent} Success Rate", color=color, linewidth=2)
        plt.fill_between(
            episodes,
            (sr_smooth - ci).clip(0, 1),
            (sr_smooth + ci).clip(0, 1),
            color=color,
            alpha=0.1
        )

    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Episode Success Rate with 95% Confidence Interval')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    # Vertical lines every k episodes to mark environment switches
    for ep in range(0, episodes.max() + 1, k):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(log_paths["plots"], "success_rate_per_episode.png")
    plt.savefig(save_path)
    plt.show()

# Example usage:
log_paths = {
    "infos": "/Users/el/dev/AIF_RedBlueDoors/logs/run_20250603_135519/infos",
    "plots": "/Users/el/dev/AIF_RedBlueDoors/logs/run_20250603_135519/plots"
}
NUM_SEEDS = 1
plot_average_episode_return_across_seeds(log_paths, NUM_SEEDS)
plot_success_rate_across_seeds(log_paths, NUM_SEEDS)
