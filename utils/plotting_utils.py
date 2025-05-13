import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_average_episode_return_across_seeds(log_paths, NUM_SEEDS, window=50):

    dfs = []
    for seed in range(NUM_SEEDS):
        df = pd.read_csv(os.path.join(log_paths["infos"], f"log_seed_{seed}.csv"))
        df['seed'] = seed
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)

    # 1️⃣ Compute total return per episode per seed (sum rewards across steps)
    episode_returns = (
        all_data.groupby(['seed', 'episode'])[['ql_reward', 'random_reward']]
        .sum()
        .reset_index()
    )

    # 2️⃣ Compute mean and SEM across seeds
    mean_returns = episode_returns.groupby('episode')[['ql_reward', 'random_reward']].mean()
    sem_returns = episode_returns.groupby('episode')[['ql_reward', 'random_reward']].sem()

    # 3️⃣ Compute 95% CI
    ci95 = 1.96 * sem_returns

    # 4️⃣ Optional smoothing
    mean_returns['ql_reward_smooth'] = mean_returns['ql_reward'].rolling(window, min_periods=1).mean()
    mean_returns['random_reward_smooth'] = mean_returns['random_reward'].rolling(window, min_periods=1).mean()

    ci95['ql_reward_smooth'] = ci95['ql_reward'].rolling(window, min_periods=1).mean()
    ci95['random_reward_smooth'] = ci95['random_reward'].rolling(window, min_periods=1).mean()

    # 5️⃣ Plot
    plt.figure(figsize=(12, 6))

    plt.plot(mean_returns.index, mean_returns['random_reward_smooth'], label='Random Mean Return', color='green')
    plt.fill_between(
        mean_returns.index,
        mean_returns['random_reward_smooth'] - ci95['random_reward_smooth'],
        mean_returns['random_reward_smooth'] + ci95['random_reward_smooth'],
        color='green', alpha=0.2
    )

    plt.plot(mean_returns.index, mean_returns['ql_reward_smooth'], label='QL Mean Return', color='orange')
    plt.fill_between(
        mean_returns.index,
        mean_returns['ql_reward_smooth'] - ci95['ql_reward_smooth'],
        mean_returns['ql_reward_smooth'] + ci95['ql_reward_smooth'],
        color='orange', alpha=0.2
    )

    plt.xlabel('Episode')
    plt.ylabel('Average Return (per episode)')
    plt.title('Average Returns per Episode with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)

    # Vertical lines every 50 episodes to mark environment switches
    for ep in range(0, mean_returns.index.max() + 1, 50):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(log_paths["plots"], "average_returns_per_episode.png"))
    plt.show()
