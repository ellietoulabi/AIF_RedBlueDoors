import argparse
import os
import csv
import random
import numpy as np
import copy
from tqdm import trange
from datetime import datetime
import wandb
import sys
sys.path.append("..")

from pymdp.agent import Agent

from envs.redbluedoors_env.ma_redbluedoors import RedBlueDoorEnv

from agents.ql import QLearningAgent

from agents.aif_models import model_2
from agents.aif_models.model_2 import convert_obs_to_active_inference_format

from utils.env_utils import get_config_path
from utils.logging_utils import create_experiment_folder
from utils.plotting_utils import plot_average_episode_return_across_seeds




def run_experiment(seed, q_table_path, log_filename, episodes=100, max_steps=150, change_every=50):
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
            "step"
            "aif_action",
            "ql_action",
            "aif_reward",
            "ql_reward",
            "q_pi",
            "G",           
        ]
        writer = csv.writer(file)
        writer.writerow(fieldnames)


    config_paths = [
        "../envs/redbluedoors_env/configs/config.json",
        "../envs/redbluedoors_env/configs/config2.json",
    ]


    for episode in trange(episodes, desc=f"Seed {seed} Training"):
        config_path = get_config_path(config_paths, episode, k=change_every, alternate=True)
        env = RedBlueDoorEnv(max_steps=max_steps, config_path=config_path)
        obs, _ = env.reset()
        aif_obs = convert_obs_to_active_inference_format(obs)
        state = ql_agent.get_state(obs)
        
        step_reward_list_aif = []
        step_reward_list_ql = []

        aif_agent.D = copy.deepcopy(model_2.MODEL["D"])

        for step in range(max_steps):

            qs = aif_agent.infer_states(aif_obs)
            # aif_agent.D = qs
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

            step_reward_aif = rewards.get("agent_0", 0)
            step_reward_ql = rewards.get("agent_1", 0)
            
            step_reward_list_aif.append(step_reward_aif)
            step_reward_list_ql.append(step_reward_ql)

            state = next_state
            wandb.log({
                "seed": seed,
                "episode": episode,
                "step":step,
                "aif_action":int(next_action_aif[0]),
                "ql_action": int(next_action_ql),
                "aif_reward": step_reward_aif,
                "ql_reward": step_reward_ql,
                "q_pi": q_pi,
                "G": G,
            })
            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        seed,
                        episode,
                        step,
                        int(next_action_aif[0]),
                        int(next_action_ql),
                        step_reward_aif,
                        step_reward_ql,
                        q_pi,
                        G,
                        
                    ]
                )

            if any(terminations.values()) or any(truncations.values()):
                qs = aif_agent.infer_states(aif_obs)
                q_pi, G = aif_agent.infer_policies()
                aif_agent.D = qs
                break

            aif_obs = convert_obs_to_active_inference_format(obs)

        ql_agent.decay_exploration()

        wandb.log({
            "seed": seed,
            "episode": episode,
            "aif_return_of_episode": np.sum(step_reward_list_aif),
            "ql_return_of_episode": np.sum(step_reward_list_ql),
            "aif_average_reward_of_episode": np.mean(step_reward_list_aif),
            "ql_average_reward_of_episode": np.mean(step_reward_list_ql),
            
        })

        env.close()

    return True

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Integer seed for this run"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes (default=100)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=150,
        help="Max steps per episode (default=150)"
    )
    
    parser.add_argument(
        "--change_every",
        type=int,
        default=50,
        help="Change map every k episodes (default=50)"
    )
    
    args = parser.parse_args()
    SEED      = args.seed
    EPISODES  = args.episodes
    MAX_STEPS = args.max_steps
    CHANGE_EVERY = args.change_every
    

    metadata = {
        "description": "Experiment comparing AIF and QL agents in the Red-Blue Doors environment.",
        "agents": ["AIF", "QL"],
        "environment": "Red-Blue Doors",
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "map_config": ["configs/config.json", "configs/config2.json"],
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "episodes": EPISODES,
        "change_every": CHANGE_EVERY,
    }
    print("Experiment metadata:", metadata)
    
    wandb.init(
        project="redbluedoors",  # you can choose your own project name
        name=f"seed_{SEED}_{datetime.now().strftime('%H%M%S')}",
        config=metadata,
        reinit=True   # allows multiple calls to wandb.init() in the same session
    )

    q_table_file = f"aif_ql_q_table_seed_{SEED}.json"
    log_file = f"aif_ql_log_seed_{SEED}.csv"
    _done = run_experiment(SEED, q_table_file, log_file, EPISODES, MAX_STEPS, CHANGE_EVERY)

    print("Experiment completed. Results saved.")
    wandb.finish()


if __name__ == "__main__":
    main()

#___________________

# log_paths = create_experiment_folder(base_dir="../logs", metadata=metadata)
# print("Logging folders:")
# for k, v in log_paths.items():
#     print(f"{k}: {v}")

# seeds = [0, 1, 2, 3, 4]  # or as many as you want
# all_results = []

# for seed in seeds:
#     q_table_file = os.path.join(log_paths["root"],f"q_table_seed_{seed}.json")
#     log_file = os.path.join(log_paths["infos"],f"log_seed_{seed}.csv")
#     rewards_aif, rewards_ql = run_experiment(seed, q_table_file, log_file, 100, 100)

#     for ep, (ra, rq) in enumerate(zip(rewards_aif, rewards_ql)):
#         all_results.append(
#             {"seed": seed, "episode": ep, "aif_reward": ra, "ql_reward": rq}
#         )



# plot_average_episode_return_across_seeds(log_paths, metadata["seeds"], window=5,agent_names=['aif_reward', 'ql_reward'])
# print("Experiment completed. Results saved.")