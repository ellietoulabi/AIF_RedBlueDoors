
import argparse
import os
import csv
import random
import numpy as np
import copy
from tqdm import trange
from datetime import datetime
# import wandb
import sys
sys.path.append("..")

from envs.redbluedoors_env.ma_redbluedoors import RedBlueDoorEnv

from utils.env_utils import get_config_path



def run_experiment(seed, log_filename, episodes=2000, max_steps=150, change_every=50):
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

    

    config_paths = [
        "../envs/redbluedoors_env/configs/config.json",
        "../envs/redbluedoors_env/configs/config2.json",
    ]
    



    for episode in trange(episodes, desc=f"Seed {seed} Training"):
        config_path = get_config_path(config_paths, episode, k=change_every, alternate=True)
        env = RedBlueDoorEnv(max_steps=max_steps, config_path=config_path)
        obs, _ = env.reset()
        
        step_reward_list_rand1 = []
        step_reward_list_rand2 = []
        
        for step in range(max_steps):
            
            step_reward_rand1 = 0
            step_reward_rand2 = 0

            next_action_rand1 = np.random.choice(len(env.ACTION_MEANING))
            next_action_rand2 = np.random.choice(len(env.ACTION_MEANING))
            action_dict = {
                "agent_0": int(next_action_rand1),
                "agent_1": int(next_action_rand2),
            }

            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            step_reward_rand1 = rewards.get("agent_0", 0)
            step_reward_rand2 = rewards.get("agent_1", 0)
            
            step_reward_list_rand1.append(step_reward_rand1)
            step_reward_list_rand2.append(step_reward_rand2)
            
            # wandb.log({
            #     "seed": seed,
            #     "episode": episode,
            #     "step":step,
            #     "rand1_action":int(next_action_rand1),
            #     "rand2_action": int(next_action_rand2),
            #     "rand1_reward": step_reward_rand1,
            #     "rand2_reward": step_reward_rand2
            # })
            
            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        seed,
                        episode,
                        step,
                        int(next_action_rand1),
                        int(next_action_rand2),
                        step_reward_rand1,
                        step_reward_rand2,
                    ]
                )

            if any(terminations.values()) or any(truncations.values()):
                break
        
        # wandb.log({
        #         "seed": seed,
        #         "episode": episode,
        #         "rand1_return_of_episode": np.sum(step_reward_list_rand1),
        #         "rand2_return_of_episode": np.sum(step_reward_list_rand2),
        #         "rand1_average_reward_of_episode": np.mean(step_reward_list_rand1),
        #         "rand2_average_reward_of_episode": np.mean(step_reward_list_rand2),
                
        #     })

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
        help="Change environment configuration every N episodes (default=50)"
    )  
    
    args = parser.parse_args()
    SEED      = args.seed
    EPISODES  = args.episodes
    MAX_STEPS = args.max_steps
    CHANGE_EVERY = args.change_every
    

    metadata = {
        "description": "Experiment comparing Random and Random agents in the Red-Blue Doors environment.",
        "agents": ["Random", "Random"],
        "environment": "Red-Blue Doors",
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "map_config": ["configs/config.json", "configs/config2.json"],
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "episodes": EPISODES,
        "change_every": CHANGE_EVERY,
    }
    
    # wandb.init(
    #     project="redbluedoors",  # you can choose your own project name
    #     name=f"seed_{SEED}_{datetime.now().strftime('%H%M%S')}",
    #     config=metadata,
    #     reinit=True   # allows multiple calls to wandb.init() in the same session
    # )
    
    # log_paths = create_experiment_folder(base_dir="../logs", metadata=metadata)
    # print("Logging folders:")
    # for k, v in log_paths.items():
    #     print(f"{k}: {v}")

    
    # q_table_file = os.path.join(log_paths["root"], f"q_table_seed_{SEED}.json")
    # log_file = os.path.join(log_paths["infos"], f"log_seed_{SEED}.csv")
    
    log_file = f"rand_rand_log_seed_{SEED}.csv"
    _done = run_experiment(SEED, log_file, EPISODES, MAX_STEPS, CHANGE_EVERY)
    
    print("Experiment completed. Results saved.")
    # wandb.finish()


if __name__ == "__main__":
    main()