import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_average_episode_return_across_seeds(
    log_paths,
    NUM_SEEDS,
    prefix,
    window=1,
    agent_names=['aif_reward', 'rand_reward'],
    k=20
):

    dfs = []
    for seed in range(NUM_SEEDS):
        fname = f"{prefix}_log_seed_{seed}.csv"
        path = os.path.join(log_paths, fname)
        df = pd.read_csv(path)
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)

    episode_returns = (
        all_data
        .groupby(['seed', 'episode'])[[agent_names[0], agent_names[1]]]
        .sum()
        .reset_index()
    )

    mean_returns = episode_returns.groupby('episode')[[agent_names[0], agent_names[1]]].mean()
    sem_returns  = episode_returns.groupby('episode')[[agent_names[0], agent_names[1]]].sem()
    ci95_returns = 1.96 * sem_returns

    for agent in agent_names:
        mean_returns[f"{agent}_smooth"]   = mean_returns[agent].rolling(window, min_periods=1).mean()
        ci95_returns[f"{agent}_smooth"]   = ci95_returns[agent].rolling(window, min_periods=1).mean()

    episodes = mean_returns.index.values
    plt.figure(figsize=(12, 6))
    for idx, agent in enumerate(agent_names):
        sm = mean_returns[f"{agent}_smooth"]
        ci = ci95_returns[f"{agent}_smooth"]
        color = f"C{idx}"
        plt.plot(episodes, sm, label=f"{agent} Return", color=color)
        plt.fill_between(
            episodes,
            np.float64(sm - ci),
            np.float64(sm + ci),
            color=color,
            alpha=0.2
        )

    plt.xlabel('Episode')
    plt.ylabel('Average Return (per episode)')
    plt.title(f'Average Returns per Episode ({prefix}) with 95% CI')
    plt.legend()
    plt.grid(True)

    for ep in range(0, int(episodes.max()) + 1, k):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(log_paths["plots"], f"{prefix}_average_returns.png")
    plt.savefig(save_path)
    plt.show()



def plot_success_rate_across_seeds(
    log_paths,
    NUM_SEEDS,
    prefix,
    window=1,
    agent_names=['aif_reward', 'rand_reward'],
    k=20
):
    
    dfs = []
    for seed in range(NUM_SEEDS):
        fname = f"{prefix}_log_seed_{seed}.csv"
        path = os.path.join(log_paths, fname)
        df = pd.read_csv(path)
        # df['seed'] = seed
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)

    episode_success = all_data.groupby(['seed', 'episode'])[agent_names].max().reset_index()

    for agent in agent_names:
        episode_success[f"{agent}_success"] = (episode_success[agent] >= 1).astype(int)

    success_rates = {
        agent: episode_success
                 .groupby('episode')[f"{agent}_success"]
                 .mean()
                 .rename(f"{agent}_rate")
        for agent in agent_names
    }
    sem_success = {
        agent: episode_success
                 .groupby('episode')[f"{agent}_success"]
                 .sem()
                 .rename(f"{agent}_sem")
        for agent in agent_names
    }
    ci95_success = {agent: 1.96 * sem_success[agent] for agent in agent_names}

    success_rate_smooth    = {}
    ci95_success_smooth    = {}
    for agent in agent_names:
        success_rate_smooth[agent] = success_rates[agent].rolling(window, min_periods=1).mean()
        ci95_success_smooth[agent] = ci95_success[agent].rolling(window, min_periods=1).mean()

    episodes = success_rates[agent_names[0]].index.values
    plt.figure(figsize=(12, 6))
    for idx, agent in enumerate(agent_names):
        sr_smooth = success_rate_smooth[agent]
        ci = ci95_success_smooth[agent]
        color = f"C{idx}"
        plt.plot(episodes, sr_smooth, label=f"{agent} Success Rate", color=color, linewidth=2)
        plt.fill_between(
            episodes,
            np.float64((sr_smooth - ci).clip(0, 1)),
            np.float64((sr_smooth + ci).clip(0, 1)),
            color=color,
            alpha=0.1
        )

    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'Episode Success Rate ({prefix}) with 95% CI')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    for ep in range(0, int(episodes.max()) + 1, k):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(log_paths["plots"], f"{prefix}_success_rate.png")
    plt.savefig(save_path)
    plt.show()



def plot_average_episode_return_across_seeds(log_paths, NUM_SEEDS, window=50, agent_names=['aif_reward', 'ql_reward'],k=20,prefix="aif_ql"):

    dfs = []
    for seed in range(NUM_SEEDS):
        fname = f"{prefix}_log_seed_{seed}.csv"
        path = os.path.join(log_paths, fname)
        df = pd.read_csv(path)
        # df = pd.read_csv(os.path.join(log_paths["infos"], f"log_seed_{seed}.csv"))
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

    # Vertical lines every k episodes to mark environment switches
    for ep in range(0, mean_returns.index.max() + 1, k):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(log_paths, "returns_per_episode.png"))
    plt.show()




def plot_episode_return_for_one_seed_csv(csv_path, agent_cols=["aif_reward", "rand_reward"], save_path=None, smooth_window=1):

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
    
    
    


# Example usage:
# log_paths = {
#     "infos": "/Users/el/dev/AIF_RedBlueDoors/logs/cc_logs",
#     "plots": "/Users/el/dev/AIF_RedBlueDoors/logs/cc_logs"
# }
log_paths="/Users/el/dev/AIF_RedBlueDoors/logs/cc_logs"
NUM_SEEDS = 5
plot_average_episode_return_across_seeds(log_paths, NUM_SEEDS, prefix="aif_ql", k=50,agent_names=['aif_reward', 'ql_reward'])
plot_success_rate_across_seeds(log_paths, NUM_SEEDS, prefix="aif_ql",k=50,agent_names=['aif_reward', 'ql_reward'])
