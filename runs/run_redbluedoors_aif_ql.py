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
from agents.ql import QLearningAgent
from agents.aif_models import model_2
from agents.aif_models.model_2 import convert_obs_to_active_inference_format
from utils.env_utils import get_config_path
from utils.logging_utils import create_experiment_folder
from utils.plotting_utils import plot_average_episode_return_across_seeds


metadata = {
    "description": "Experiment comparing AIF and Q-learning agents in the Red-Blue Doors environment.",
    "agents": ["AIF", "Q-learning"],
    "environment": "Red-Blue Doors",
    "date": "2024-04-28",
    "map_config": ["configs/config.json", "configs/config2.json"],
    "seeds": 5,
    "max_steps": 150,
    "episodes": 100
    
}

log_paths = create_experiment_folder(base_dir="../logs", metadata=metadata)
print("Logging folders:")
for k, v in log_paths.items():
    print(f"{k}: {v}")

def run_experiment(seed, q_table_path, log_filename, episodes=100, max_steps=150):
    np.random.seed(seed)
    random.seed(seed)

    # Re-create the agents fresh for each seed
    aif_agent = Agent(
        A=model_2.MODEL["A"],
        B=model_2.MODEL["B"],
        C=model_2.MODEL["C"],
        D=model_2.MODEL["D"],
        pA=model_2.MODEL["pA"],
        inference_algo="MMP",
        policy_len=2,
        inference_horizon=2,
        sampling_mode="marginal",
        action_selection="stochastic",
        alpha=0.1,
    )

    ql_agent = QLearningAgent(
        agent_id="agent_1",
        action_space_size=5,
        q_table_path=q_table_path,
        load_existing=False,
    )

    # Logging
    with open(log_filename, mode="w", newline="") as file:
        fieldnames = [
            "seed",
            "episode",
            "step",
            "ql_action",
            "aif_action",
            "ql_reward",
            "aif_reward",
        ]
        writer = csv.writer(file)
        writer.writerow(fieldnames)

    EPISODES = episodes
    MAX_STEPS = max_steps

    config_paths = [
        "../envs/redbluedoors_env/configs/config.json",
        "../envs/redbluedoors_env/configs/config2.json",
    ]

    reward_log_aif = []
    reward_log_ql = []

    for episode in trange(EPISODES, desc=f"Seed {seed} Training"):
        config_path = get_config_path(config_paths, episode)
        env = RedBlueDoorEnv(max_steps=MAX_STEPS, config_path=config_path)
        obs, _ = env.reset()
        aif_obs = convert_obs_to_active_inference_format(obs)
        state = ql_agent.get_state(obs)
        total_reward_aif = 0
        total_reward_ql = 0

        aif_agent.D = copy.deepcopy(model_2.MODEL["D"])

        for step in range(MAX_STEPS):

            qs = aif_agent.infer_states(aif_obs)
            aif_agent.D = qs
            q_pi, G = aif_agent.infer_policies()

            next_action_aif = aif_agent.sample_action()
            next_action_ql = ql_agent.choose_action(state, "agent_1")

            action_dict = {
                "agent_0": int(next_action_aif[0]),
                "agent_1": int(next_action_ql),
            }

            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            next_state = ql_agent.get_state(obs)

            if rewards and next_state is not None:
                ql_agent.update_q_table(
                    state, action_dict, rewards, next_state, "agent_1"
                )

            total_reward_aif += rewards.get("agent_0", 0)
            total_reward_ql += rewards.get("agent_1", 0)

            state = next_state

            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        seed,
                        episode,
                        step,
                        int(next_action_ql),
                        int(next_action_aif[0]),
                        rewards.get("agent_1", 0),
                        rewards.get("agent_0", 0),
                    ]
                )

            if any(terminations.values()) or any(truncations.values()):
                break

            aif_obs = convert_obs_to_active_inference_format(obs)

        ql_agent.decay_exploration()

        reward_log_aif.append(total_reward_aif)
        reward_log_ql.append(total_reward_ql)

        env.close()

    return reward_log_aif, reward_log_ql


seeds = [0, 1, 2, 3, 4]  # or as many as you want
all_results = []

for seed in seeds:
    q_table_file = os.path.join(log_paths["root"],f"q_table_seed_{seed}.json")
    log_file = os.path.join(log_paths["infos"],f"log_seed_{seed}.csv")
    rewards_aif, rewards_ql = run_experiment(seed, q_table_file, log_file, 2000, 100)

    for ep, (ra, rq) in enumerate(zip(rewards_aif, rewards_ql)):
        all_results.append(
            {"seed": seed, "episode": ep, "aif_reward": ra, "ql_reward": rq}
        )



plot_average_episode_return_across_seeds(log_paths, metadata["seeds"], window=10,agent_names=['aif_reward', 'ql_reward'])
print("Experiment completed. Results saved.")