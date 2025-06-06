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


def run_experiment(seed, q_table_path, log_filename, episodes=2000, max_steps=150, change_every=50):
    np.random.seed(seed)
    random.seed(seed)

    # Re-create the agents fresh for each seed

    ql_agent = QLearningAgent(
        agent_id="agent_1",
        action_space_size=5,
        q_table_path=q_table_path,
        load_existing=False    )

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
        
        state = ql_agent.get_state(obs)
        step_reward_list_rand = []
        step_reward_list_ql = []

        for step in range(max_steps):
            
            step_reward_rand = 0
            step_reward_ql = 0

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

            step_reward_rand = rewards.get("agent_0", 0)
            step_reward_ql = rewards.get("agent_1", 0)
            
            step_reward_list_rand.append(step_reward_rand)
            step_reward_list_ql.append(step_reward_ql)

            state = next_state
            # wandb.log({
            #     "seed": seed,
            #     "episode": episode,
            #     "step":step,
            #     "random_action":int(next_action_random),
            #     "ql_action": int(next_action_ql),
            #     "random_reward": step_reward_rand,
            #     "ql_reward": step_reward_ql
            # })
            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    seed,
                    episode,
                    step,
                    int(next_action_random),
                    int(next_action_ql),
                    step_reward_rand,
                    step_reward_ql,
                ])
            # print(f"QL Action: {next_action_ql}, Random Action: {next_action_random}")
            # print(f"QTable:")
            # for k, v in ql_agent.q_table.items():
            #     print(f"State: {k}, Q-values: {v}")
            # print("____" * 20)
            if any(terminations.values()) or any(truncations.values()):
                break


        ql_agent.decay_exploration()

        # wandb.log({
        #     "seed": seed,
        #     "episode": episode,
        #     "rand_return_of_episode": np.sum(step_reward_list_rand),
        #     "ql_return_of_episode": np.sum(step_reward_list_ql),
        #     "rand_average_reward_of_episode": np.mean(step_reward_list_rand),
        #     "ql_average_reward_of_episode": np.mean(step_reward_list_ql),
            
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
        "description": "Experiment comparing Random and QL agents in the Red-Blue Doors environment.",
        "agents": ["Random", "QL"],
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

    q_table_file = f"ql_rand_q_table_seed_{SEED}.json"
    log_file = f"ql_rand_log_seed_{SEED}.csv"
    _done = run_experiment(SEED, q_table_file, log_file, EPISODES, MAX_STEPS, CHANGE_EVERY)

    print("Experiment completed. Results saved.")
    # wandb.finish()


if __name__ == "__main__":
    main()