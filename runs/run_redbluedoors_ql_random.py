
import copy
import csv
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


from envs.redbluedoors_env.ma_redbluedoors_env import RedBlueDoorEnv
from agents.ql import QLearningAgent
from agents.random import RandomAgent
from utils.env_utils import get_config_path
from utils.logging_utils import create_experiment_folder



metadata = {
    "agents": "Random + Q-Learning",
    "env": "RedBlueDoors",
    "map_config": ["configs/config.json", "configs/config2.json"],
    "seeds": 5,
    "max_steps": 150,
    "episodes": 100,
    "description": "Random agent with Q-learning agent in RedBlueDoors environment",
}

log_paths = create_experiment_folder(base_dir="logs", metadata=metadata)
print("Logging folders:")
for k, v in log_paths.items():
    print(f"{k}: {v}")

# Initialize wandb




def run_experiment(seed, q_table_path, log_filename, episodes=2000, max_steps=150):
    np.random.seed(seed)
    random.seed(seed)

    # Re-create the agents fresh for each seed
    random_agent = RandomAgent()

    ql_agent = QLearningAgent(
        agent_id="agent_1",
        action_space_size=5,
        q_table_path=q_table_path,
        load_existing=False,
        epsilon_decay=0.99
    )

    # Logging
    with open(log_filename, mode="w", newline="") as file:
        fieldnames = [
            "seed",
            "episode",
            "step",
            "ql_action",
            "random_action",
            "ql_reward",
            "random_reward",
            "map",
        ]
        writer = csv.writer(file)
        writer.writerow(fieldnames)

    EPISODES = episodes
    MAX_STEPS = max_steps

    config_paths = [
        "envs/redbluedoors_env/configs/config.json",
        "envs/redbluedoors_env/configs/config2.json",
    ]

    reward_log_random = []
    reward_log_ql = []

    for episode in trange(EPISODES, desc=f"Seed {seed} Training"):
        config_path = get_config_path(config_paths, episode)
        env = RedBlueDoorEnv(max_steps=MAX_STEPS, config_path=config_path)
        obs, _ = env.reset()
        state = ql_agent.get_state(obs)
        total_reward_random = 0
        total_reward_ql = 0

        for step in range(MAX_STEPS):

            
            next_action_random = random_agent.choose_action()
            next_action_ql = ql_agent.choose_action(state, "agent_1")

            action_dict = {
                "agent_0": int(next_action_random),
                "agent_1": int(next_action_ql),
            }

            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            next_state = ql_agent.get_state(obs)

            if rewards and next_state is not None:
                ql_agent.update_q_table(state, action_dict, rewards, next_state, "agent_1")

            total_reward_random += rewards.get("agent_0", 0)
            total_reward_ql += rewards.get("agent_1", 0)

            state = next_state

            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    seed,
                    episode,
                    step,
                    int(next_action_ql),
                    int(next_action_random),
                    rewards.get("agent_1", 0),
                    rewards.get("agent_0", 0),
                    env.render()
                ])
            
            if episode % 100 == 0 and step == 0:
                print(f"Unique states visited: {len(ql_agent.q_table)}")
            if any(terminations.values()) or any(truncations.values()):
                break


        ql_agent.decay_exploration()

        reward_log_random.append(total_reward_random)
        reward_log_ql.append(total_reward_ql)

        env.close()

    return reward_log_random, reward_log_ql

NUM_SEEDS = 5  # update this to however many seeds you actually have
seeds = [0, 1, 2, 3, 4]  # or as many as you want
all_results = []

for seed in seeds:
    q_table_file = os.path.join(log_paths["infos"],f"q_table_seed_{seed}.json")
    log_file = os.path.join(log_paths["infos"],f"log_seed_{seed}.csv")
    rewards_aif, rewards_ql = run_experiment(seed, q_table_file, log_file, 2000, 150)

    for ep, (ra, rq) in enumerate(zip(rewards_aif, rewards_ql)):
        all_results.append({"seed": seed, "episode": ep, "random_reward": ra, "ql_reward": rq})



# --- LOAD DATA ---co

dfs = []
for seed in range(NUM_SEEDS):
    df = pd.read_csv(os.path.join(log_paths["infos"],f"log_seed_{seed}.csv"))
    df['seed'] = seed
    dfs.append(df)

all_data = pd.concat(dfs)



# # --- GROUP BY EPISODE ---

mean_df = all_data.groupby('episode')[['ql_reward', 'random_reward']].mean()
sem_df = all_data.groupby('episode')[['ql_reward', 'random_reward']].sem()

# # --- 95% CONFIDENCE INTERVAL (CI) ---

ci95 = 1.96 * sem_df / np.sqrt(NUM_SEEDS)

# # --- SMOOTH (optional) ---

window = 50

mean_df['ql_reward_smooth'] = mean_df['ql_reward'].rolling(window, min_periods=1).mean()
mean_df['random_reward_smooth'] = mean_df['random_reward'].rolling(window, min_periods=1).mean()

ci95['ql_reward_smooth'] = ci95['ql_reward'].rolling(window, min_periods=1).mean()
ci95['random_reward_smooth'] = ci95['random_reward'].rolling(window, min_periods=1).mean()

# # --- PLOT ---

plt.figure(figsize=(12, 6))

# AIF agent
plt.plot(mean_df.index, mean_df['random_reward_smooth'] , label='Random Mean Reward', color='green')
plt.fill_between(mean_df.index,
                 np.float64(mean_df['random_reward_smooth'] - ci95['random_reward_smooth']) ,
                 np.float64(mean_df['random_reward_smooth'] + ci95['random_reward_smooth']) ,
                 color='green', alpha=0.2)

# QL agent
plt.plot(mean_df.index, mean_df['ql_reward_smooth'], label='QL Mean Reward', color='orange')
plt.fill_between(mean_df.index,
                 np.float64(mean_df['ql_reward_smooth'] - ci95['ql_reward_smooth']),
                 np.float64(mean_df['ql_reward_smooth'] + ci95['ql_reward_smooth']),
                 color='orange', alpha=0.2)


plt.xlabel('Episode')
plt.ylabel('Smoothed Reward')
plt.title('Average Rewards per Episode with 95% Confidence Interval')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_paths["plots"], "average_rewards_per_episode.png"))