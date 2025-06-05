
import argparse
import os
import csv
import random
import numpy as np
import copy
from tqdm import trange
from datetime import datetime
import sys
sys.path.append("..")

from pymdp.agent import Agent

from envs.redbluedoors_env.ma_redbluedoors import RedBlueDoorEnv

from utils.env_utils import get_config_path

from agents.aif_models import model_2
from agents.aif_models.model_2 import convert_obs_to_active_inference_format


def run_experiment(seed, log_filename, episodes=2000, max_steps=150, change_every=50):
    print(np.random.seed(seed))
    print(random.seed(seed))
    

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
        gamma=4.0,
    )

    # Logging
    with open(log_filename, mode="w", newline="") as file:
        fieldnames = [
            "seed",
            "episode",
            "step",
            "aif_action",
            "rand_action",
            "aif_reward",
            "rand_reward",
            # "qs",
            "q_pi",
            "G",
        ]
        writer = csv.writer(file)
        writer.writerow(fieldnames)

    

    config_paths = [
        "../envs/redbluedoors_env/configs/config.json",
        "../envs/redbluedoors_env/configs/config2.json",
    ]

    reward_log_aif = []
    reward_log_rand = []

    for episode in trange(episodes, desc=f"Seed {seed} Training"):
        config_path = get_config_path(config_paths, episode, k=change_every, alternate=True)
        env = RedBlueDoorEnv(max_steps=max_steps, config_path=config_path)
        obs, _ = env.reset()
        aif_obs = convert_obs_to_active_inference_format(obs)
        total_reward_aif = 0
        total_reward_rand = 0

        aif_agent.D = copy.deepcopy(model_2.MODEL["D"])

        for step in range(max_steps):

            qs = aif_agent.infer_states(aif_obs)
            aif_agent.D = qs
            q_pi, G = aif_agent.infer_policies()
            

            next_action_aif = aif_agent.sample_action()
            next_action_rand = np.random.choice(len(env.ACTION_MEANING))

            action_dict = {
                "agent_0": int(next_action_aif[0]),
                "agent_1": int(next_action_rand),
            }

            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            total_reward_aif += rewards.get("agent_0", 0)
            total_reward_rand += rewards.get("agent_1", 0)

            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        seed,
                        episode,
                        step,
                        int(next_action_aif[0]),
                        int(next_action_rand),
                        rewards.get("agent_0", 0),
                        rewards.get("agent_1", 0),
                        # qs,
                        q_pi,
                        G,
                    ]
                )
                
            aif_obs = convert_obs_to_active_inference_format(obs)
         
            # debug_plot_inference_step(
            #     qs=qs,
            #     q_pi=q_pi,
            #     G=G,
            #     step=step,
            #     episode=episode,
            #     actual_state=aif_obs,
            #     action_taken=int(next_action_aif[0]),
            #     save_dir="./debugplots"  # or None to just show
            # )

            if any(terminations.values()) or any(truncations.values()):
                qs = aif_agent.infer_states(aif_obs)
                q_pi, G = aif_agent.infer_policies()
                aif_agent.D = qs
                # aif_agent.G = G
                # aif_agent.qs = qs
                # aif_agent.q_pi = q_pi
                break


        reward_log_aif.append(total_reward_aif)
        reward_log_rand.append(total_reward_rand)

        env.close()

    return reward_log_aif, reward_log_rand



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
        "description": "Experiment comparing AIF and Random agents in the Red-Blue Doors environment.",
        "agents": ["AIF", "Random"],
        "environment": "Red-Blue Doors",
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "map_config": ["configs/config.json", "configs/config2.json"],
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "episodes": EPISODES,
        "change_every": CHANGE_EVERY,
    }

    # log_paths = create_experiment_folder(base_dir="../logs", metadata=metadata)
    # print("Logging folders:")
    # for k, v in log_paths.items():
    #     print(f"{k}: {v}")


    all_results = []
    
    # q_table_file = os.path.join(log_paths["root"], f"q_table_seed_{SEED}.json")
    # log_file = os.path.join(log_paths["infos"], f"log_seed_{SEED}.csv")
    
    
    log_file = f"aif_rand_log_seed_{SEED}.csv"
        
    rewards_aif, rewards_rand = run_experiment(SEED, log_file, EPISODES, MAX_STEPS, CHANGE_EVERY)



    print("Experiment completed. Results saved.")


if __name__ == "__main__":
    main()