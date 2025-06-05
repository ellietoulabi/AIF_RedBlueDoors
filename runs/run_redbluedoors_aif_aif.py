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

from agents.aif_models import model_2
from agents.aif_models import model_3

from agents.aif_models.model_2 import convert_obs_to_active_inference_format

from utils.env_utils import get_config_path
from utils.logging_utils import create_experiment_folder
from utils.plotting_utils import plot_average_episode_return_across_seeds




def run_experiment(seed, log_filename, episodes=100, max_steps=150, change_every=50):
    np.random.seed(seed)
    random.seed(seed)

    # Re-create the agents fresh for each seed
    aif_agent1 = Agent(
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
        gamma=4.0,
    )

    aif_agent2 = Agent(
        A=model_3.MODEL["A"],
        B=model_3.MODEL["B"],
        C=model_3.MODEL["C"],
        D=model_3.MODEL["D"],
        pA=model_3.MODEL["pA"],
        inference_algo="MMP",
        policy_len=2,
        inference_horizon=2,
        sampling_mode="marginal",
        action_selection="stochastic",
        alpha=0.1,
        gamma=4.0,
    )

    # Logging
    with open(log_filename, mode="w", newline="") as file:
        fieldnames = [
            "seed",
            "episode",
            "step"
            "aif1_action",
            "aif2_action",
            "aif1_reward",
            "aif2_reward",
            "aif1_q_pi",
            "aif2_q_pi",
            "aif1_G",
            "aif2_G",
           
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
        aif1_obs = convert_obs_to_active_inference_format(obs, "agent_0")
        aif2_obs = convert_obs_to_active_inference_format(obs, "agent_1")
        
        step_reward_list_aif1 = []
        step_reward_list_aif2 = []

        aif_agent1.D = copy.deepcopy(model_2.MODEL["D"])
        aif_agent2.D = copy.deepcopy(model_3.MODEL["D"])

        for step in range(max_steps):

            qs1 = aif_agent1.infer_states(aif1_obs)
            q_pi1, G1 = aif_agent1.infer_policies()
            
            
            qs2 = aif_agent2.infer_states(aif2_obs)
            q_pi2, G2 = aif_agent2.infer_policies()
            # aif_agent.D = qs
            

            next_action_aif1 = aif_agent1.sample_action()
            next_action_aif2 = aif_agent2.sample_action()

            action_dict = {
                "agent_0": int(next_action_aif1[0]),
                "agent_1": int(next_action_aif2[0]),
            }

            obs, rewards, terminations, truncations, infos = env.step(action_dict)

        

            step_reward_aif1 = rewards.get("agent_0", 0)
            step_reward_aif2 = rewards.get("agent_1", 0)
            
            step_reward_list_aif1.append(step_reward_aif1)
            step_reward_list_aif2.append(step_reward_aif2)

            wandb.log({
                "seed": seed,
                "episode": episode,
                "step":step,
                "aif1_action":int(next_action_aif1[0]),
                "aif2_action": int(next_action_aif2[0]),
                "aif1_reward": step_reward_aif1,
                "aif2_reward": step_reward_aif2,
                "aif1_q_pi": q_pi1,
                "aif2_q_pi": q_pi2,
                "aif1_G": G1,
                "aif2_G": G2,
            })
            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        seed,
                        episode,
                        step,
                        int(next_action_aif1[0]),
                        int(next_action_aif2[0]),
                        step_reward_aif1,
                        step_reward_aif2,
                        q_pi1,
                        q_pi2,
                        G1,
                        G2,
                        
                    ]
                )
            
            aif1_obs = convert_obs_to_active_inference_format(obs, "agent_0")
            aif2_obs = convert_obs_to_active_inference_format(obs, "agent_1")

            if any(terminations.values()) or any(truncations.values()):
                qs1 = aif_agent1.infer_states(aif1_obs)
                q_pi1, G1 = aif_agent1.infer_policies()
                
                qs2 = aif_agent2.infer_states(aif2_obs)
                q_pi2, G2 = aif_agent2.infer_policies()
                # aif_agent.D = qs
                break

            

        wandb.log({
            "seed": seed,
            "episode": episode,
            "aif1_return_of_episode": np.sum(step_reward_list_aif1),
            "aif2_return_of_episode": np.sum(step_reward_list_aif2),
            "aif1_average_reward_of_episode": np.mean(step_reward_list_aif1),
            "aif2_average_reward_of_episode": np.mean(step_reward_list_aif2),
            
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
        "description": "Experiment comparing AIF and AIF agents in the Red-Blue Doors environment.",
        "agents": ["AIF", "AIF"],
        "environment": "Red-Blue Doors",
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "map_config": ["configs/config.json", "configs/config2.json"],
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "episodes": EPISODES,
        "change_every": CHANGE_EVERY,
    }
    print("Metadata for this run:")
    for k, v in metadata.items():
        print(f"{k}: {v}")
    
    wandb.init(
        project="redbluedoors",  # you can choose your own project name
        name=f"seed_{SEED}_{datetime.now().strftime('%H%M%S')}",
        config=metadata,
        reinit=True   # allows multiple calls to wandb.init() in the same session
    )

    log_file = f"aif_aif_log_seed_{SEED}.csv"
    _done = run_experiment(SEED, log_file, EPISODES, MAX_STEPS, CHANGE_EVERY)

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


