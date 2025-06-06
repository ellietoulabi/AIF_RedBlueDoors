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

from agents.ql import QLearningAgent


def run_experiment(seed, q_table_paths, log_filename, episodes=2000, max_steps=150, change_every=50):
    np.random.seed(seed)
    random.seed(seed)

    # Re-create the agents fresh for each seed

    ql_agent1 = QLearningAgent(
        agent_id="agent_0",
        action_space_size=5,
        q_table_path=q_table_paths[0],
        load_existing=False    )
    
    ql_agent2 = QLearningAgent(
        agent_id="agent_1",
        action_space_size=5,
        q_table_path=q_table_paths[1],
        load_existing=False    )

    # Logging
    with open(log_filename, mode="w", newline="") as file:
        fieldnames = [
            "seed",
            "episode",
            "step",
            "ql1_action",
            "ql2_action",
            "ql1_reward",
            "ql2_reward",
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
        
        ql1_state = ql_agent1.get_state(obs)
        ql2_state = ql_agent2.get_state(obs)
        
        step_reward_list_ql1 = []
        step_reward_list_ql2 = []

        for step in range(max_steps):
            
            step_reward_rand = 0
            step_reward_ql = 0

            next_action_ql1 = ql_agent1.choose_action(ql1_state, "agent_0")
            next_action_ql2 = ql_agent2.choose_action(ql2_state, "agent_1")
        
            action_dict = {
                "agent_0": int(next_action_ql1),
                "agent_1": int(next_action_ql2),
            }

            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            ql1_next_state = ql_agent1.get_state(obs)
            ql2_next_state = ql_agent2.get_state(obs)
            

            if rewards and ql1_next_state is not None and ql2_next_state is not None:
                ql_agent1.update_q_table(ql1_state, action_dict, rewards, ql1_next_state, "agent_0")
                ql_agent2.update_q_table(ql2_state, action_dict, rewards, ql2_next_state, "agent_1")

            step_reward_ql1 = rewards.get("agent_0", 0)
            step_reward_ql2 = rewards.get("agent_1", 0)
            
            step_reward_list_ql1.append(step_reward_ql1)
            step_reward_list_ql2.append(step_reward_ql2)

            ql1_state = ql1_next_state
            ql2_state = ql2_next_state
            
            # wandb.log({
            #     "seed": seed,
            #     "episode": episode,
            #     "step":step,
            #     "random_action":int(next_action_ql1),
            #     "ql_action": int(next_action_ql2),
            #     "random_reward": step_reward_ql1,
            #     "ql_reward": step_reward_ql2
            # })
            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    seed,
                    episode,
                    step,
                    int(next_action_ql1),
                    int(next_action_ql2),
                    step_reward_ql1,
                    step_reward_ql2,
                ])
            # print(f"QL Action: {next_action_ql}, Random Action: {next_action_random}")
            # print(f"QTable:")
            # for k, v in ql_agent.q_table.items():
            #     print(f"State: {k}, Q-values: {v}")
            # print("____" * 20)
            if any(terminations.values()) or any(truncations.values()):
                break


        ql_agent1.decay_exploration()
        ql_agent2.decay_exploration()
        

        # wandb.log({
        #     "seed": seed,
        #     "episode": episode,
        #     "rand_return_of_episode": np.sum(step_reward_list_ql1),
        #     "ql_return_of_episode": np.sum(step_reward_list_ql2),
        #     "rand_average_reward_of_episode": np.mean(step_reward_list_ql1),
        #     "ql_average_reward_of_episode": np.mean(step_reward_list_ql2),
            
        # })

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
        "description": "Experiment comparing QL and QL agents in the Red-Blue Doors environment.",
        "agents": ["QL", "QL"],
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

    q_table_file = [f"ql_ql_q_table1_seed_{SEED}.json", f"ql_ql_q_table2_seed_{SEED}.json"]
    log_file = f"ql_ql_log_seed_{SEED}.csv"
    _done = run_experiment(SEED, q_table_file, log_file, EPISODES, MAX_STEPS, CHANGE_EVERY)

    print("Experiment completed. Results saved.")
    # wandb.finish()


if __name__ == "__main__":
    main()