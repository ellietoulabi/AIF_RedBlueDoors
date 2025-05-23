import os
import csv
import random
import numpy as np
import copy
from tqdm import trange
import sys

from pymdp.agent import Agent

sys.path.append("..")
from envs.redbluedoors_env.ma_redbluedoors import RedBlueDoorEnv
from utils.env_utils import get_config_path
from utils.logging_utils import create_experiment_folder
from utils.plotting_utils import plot_average_episode_return_across_seeds


metadata = {
    "description": "Experiment comparing Random and Random agents in the Red-Blue Doors environment.",
    "agents": ["Random", "Random"],
    "environment": "Red-Blue Doors",
    "date": "2024-04-28",
    "map_config": ["configs/config.json", "configs/config2.json"],
    "seeds": 5,
    "max_steps": 100,
    "episodes": 100
    
}
# log_paths = {"root": "../logs/run_20250513_120459",
# "plots": "../logs/run_20250513_120459/plots",
# "infos": "../logs/run_20250513_120459/infos"}
log_paths = create_experiment_folder(base_dir="../logs", metadata=metadata)
print("Logging folders:")
for k, v in log_paths.items():
    print(f"{k}: {v}")

def run_experiment(seed, q_table_path, log_filename, episodes=100, max_steps=150):
    np.random.seed(seed)
    random.seed(seed)

    # Re-create the agents fresh for each seed
    

    # Logging
    with open(log_filename, mode="w", newline="") as file:
        fieldnames = [
            "seed",
            "episode",
            "step",
            "rand1_action",
            "rand2_action",
            "rand1_reward",
            "rand2_reward",
        ]
        writer = csv.writer(file)
        writer.writerow(fieldnames)

    EPISODES = episodes
    MAX_STEPS = max_steps

    config_paths = [
        "../envs/redbluedoors_env/configs/config.json",
        "../envs/redbluedoors_env/configs/config2.json",
    ]

    reward_log_rand1 = []
    reward_log_rand2 = []

    for episode in trange(EPISODES, desc=f"Seed {seed} Training"):
        config_path = get_config_path(config_paths, episode)
        env = RedBlueDoorEnv(max_steps=MAX_STEPS, config_path=config_path)
        obs, _ = env.reset()
        total_reward_rand1 = 0
        total_reward_rand2 = 0


        for step in range(MAX_STEPS):

            

            next_action_rand1 = np.random.choice(len(env.ACTION_MEANING))
            next_action_rand2 = np.random.choice(len(env.ACTION_MEANING))
            action_dict = {
                "agent_0": int(next_action_rand1),
                "agent_1": int(next_action_rand2),
            }

            obs, rewards, terminations, truncations, infos = env.step(action_dict)


            total_reward_rand1 += rewards.get("agent_0", 0)
            total_reward_rand2 += rewards.get("agent_1", 0)

            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        seed,
                        episode,
                        step,
                        int(next_action_rand1),
                        int(next_action_rand2),
                        rewards.get("agent_0", 0),
                        rewards.get("agent_1", 0),
                    ]
                )

            if any(terminations.values()) or any(truncations.values()):
                break


        reward_log_rand1.append(total_reward_rand1)
        reward_log_rand2.append(total_reward_rand2)

        env.close()

    return reward_log_rand1, reward_log_rand2


seeds = [0, 1, 2, 3, 4]  # or as many as you want
all_results = []

for seed in seeds:
    q_table_file = os.path.join(log_paths["root"],f"q_table_seed_{seed}.json")
    log_file = os.path.join(log_paths["infos"],f"log_seed_{seed}.csv")
    rewards_rand1, rewards_rand2 = run_experiment(seed, q_table_file, log_file, 1000, 100)

    for ep, (ra, rq) in enumerate(zip(rewards_rand1, rewards_rand2)):
        all_results.append(
            {"seed": seed, "episode": ep, "rand1_reward": ra, "rand2_reward": rq}
        )



plot_average_episode_return_across_seeds(log_paths, metadata["seeds"], window=5,agent_names=['rand1_reward', 'rand2_reward'])
print("Experiment completed. Results saved.")