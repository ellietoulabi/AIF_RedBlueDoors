
import copy
import csv
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import sys


sys.path.append("..")
from envs.redbluedoors_env.ma_redbluedoors import RedBlueDoorEnv
from agents.ql import QLearningAgent
from utils.env_utils import get_config_path
from utils.logging_utils import create_experiment_folder
from utils.plotting_utils import plot_average_episode_return_across_seeds




metadata = {
    "description": "Experiment comparing Random and QLearning agents in the Red-Blue Doors environment.",
    "agents": ["Random", "QLearning"],
    "environment": "Red-Blue Doors",
    "date": "2024-04-28",
    "map_config": ["configs/config.json", "configs/config2.json"],
    "seeds": 5,
    "max_steps": 150,
    "episodes": 100
    
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
            "random_action",
            "ql_action",
            "random_reward",
            "ql_reward",
            "map",
        ]
        writer = csv.writer(file)
        writer.writerow(fieldnames)

    EPISODES = episodes
    MAX_STEPS = max_steps

    config_paths = [
        "../envs/redbluedoors_env/configs/config.json",
        "../envs/redbluedoors_env/configs/config2.json",
    ]

    reward_log_random = []
    reward_log_ql = []

    for episode in trange(EPISODES, desc=f"Seed {seed} Training"):
        config_path = get_config_path(config_paths, episode,20)
        env = RedBlueDoorEnv(max_steps=MAX_STEPS, config_path=config_path)
        obs, _ = env.reset()
        state = ql_agent.get_state(obs)
        total_reward_random = 0
        total_reward_ql = 0

        for step in range(MAX_STEPS):

            
            next_action_random = np.random.choice(len(env.ACTION_MEANING))
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
                    int(next_action_random),
                    int(next_action_ql),
                    rewards.get("agent_0", 0),
                    rewards.get("agent_1", 0),
                    env.render()
                ])
            # print(f"QL Action: {next_action_ql}, Random Action: {next_action_random}")
            # print(f"QTable:")
            # for k, v in ql_agent.q_table.items():
            #     print(f"State: {k}, Q-values: {v}")
            # print("____" * 20)
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


plot_average_episode_return_across_seeds(log_paths, metadata["seeds"], window=5,agent_names=['random_reward', 'ql_reward'],k=20)
print("Experiment completed. Results saved.")