import os
import csv
import random
import numpy as np
import copy
from tqdm import trange
import sys
import matplotlib.pyplot as plt
import time

from pymdp.agent import Agent

sys.path.append("..")
from envs.redbluedoors_env.ma_redbluedoors import RedBlueDoorEnv
from agents.aif_models import model_5
from agents.aif_models.model_5 import D1, convert_obs_to_active_inference_format
from utils.env_utils import get_config_path
from utils.logging_utils import create_experiment_folder


log_paths = create_experiment_folder(base_dir=".")
print("Logging folders:")
for k, v in log_paths.items():
    print(f"{k}: {v}")

# Main script - simple single agent AIF loop
if __name__ == "__main__":
    # Parameters
    seed = 1
    episodes =50
    max_steps = 150
    
    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)

    print(f"Running single agent AIF with seed {seed}")
    
    # Create AIF agent
    aif_agent = Agent(
        A=model_5.MODEL["A"],
        B=model_5.MODEL["B"],
        C=model_5.MODEL["C"],
        D=model_5.MODEL["D"],
        # pA=model_5.MODEL["pA"],
        inference_algo="VANILLA",
        policy_len=1,
        inference_horizon=1,
        sampling_mode="full",
        action_selection="stochastic",
        alpha=0.1,
        gamma=16.0,
        # policies= model_5.MODEL["Policies"],
        # use_states_info_gain=False

    )
    

    # Simple logging
    log_filename = f"single_agent_aif_seed_{seed}.csv"
    with open(log_filename, mode="w", newline="") as file:
        fieldnames = ["episode", "step", "action", "reward", "total_reward"]
        writer = csv.writer(file)
        writer.writerow(fieldnames)


    
    episode_rewards = []
    all_step_rewards = []  # Store step rewards for each episode

    for episode in trange(episodes, desc=f"Episode"):
        # Get config for this episode
        config_path =   "../envs/redbluedoors_env/configs/config.json"
        env = RedBlueDoorEnv(max_steps=max_steps, config_path=config_path)
        
        # Reset environment
        obs, _ = env.reset()
        # obs, rewards, terminations, truncations, infos = env.step({"agent_0": 3, "agent_1": 1}) # Initial action to start the episode
        # obs, rewards, terminations, truncations, infos = env.step({"agent_0": 3, "agent_1": 1}) # Initial action to start the episode

        # print(obs)
        
        aif_obs = convert_obs_to_active_inference_format(obs, "agent_0")
        # print(aif_obs)
        total_reward = 0
        step_rewards = []  # Store rewards for this episode
        
        
        

        for step in range(max_steps):
            # AIF inference loop
            # print("Infer state start:", time.time())
            qs = aif_agent.infer_states(aif_obs)
            # print("Infer state end:", time.time())

            # aif_agent.D = qs
            # print("Infer policies start:", time.time())
            q_pi, G = aif_agent.infer_policies()
            # print("Infer policies end:", time.time())

            # Sample action
            # print("Sample action start:", time.time())
            action = aif_agent.sample_action()
            # print("Sample action end:", time.time())
            # print(f"Episode {episode}, Step {step}, Action: {action}")
            # print(f"Q_pi: {q_pi}")
            
            action_int = int(action[0])
            # print(qs.shape)
            
            
            for f in range(len(qs)):
                p = np.argmax(qs[f])
                print(f"Factor {f}, Max Posterior: {p}")
            from pymdp.utils import plot_beliefs
            # plot_beliefs(q_pi, title=f"Episode {episode}, Step {step} - Q_pi")
            print("action:", action_int)
            
            
            # Take action (single agent)
            action_dict = {"agent_0": action_int, "agent_1": 5}  # Agent 1 does nothing
            obs, rewards, terminations, truncations, infos = env.step(action_dict)
            print(infos["map"])
            # Get reward
            reward = rewards.get("agent_0", 0)
            total_reward += reward
            step_rewards.append(reward)  # Collect step reward

            # Log step
            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([episode, step, action_int, reward, total_reward])

            # Check if episode is done
            if any(terminations.values()) or any(truncations.values()):
               
                break

            # Update observation for next step
            aif_obs = convert_obs_to_active_inference_format(obs, "agent_0")

        episode_rewards.append(total_reward)
        all_step_rewards.append(step_rewards)
        env.close()

        
    
    # Final results
    print(f"Final results:")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best episode: {np.max(episode_rewards):.2f}")
    print(f"Worst episode: {np.min(episode_rewards):.2f}")

    # Plot all step rewards concatenated as a single line
    flat_rewards = []
    episode_ends = []
    for rewards in all_step_rewards:
        flat_rewards.extend(rewards)
        episode_ends.append(len(flat_rewards))  # Mark the end of each episode

    plt.figure(figsize=(12, 6))
    plt.plot(flat_rewards, label='Step Reward (all episodes)')
    for ep_end in episode_ends[:-1]:  # Don't draw at the very end
        plt.axvline(ep_end, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Step (across all episodes)')
    plt.ylabel('Reward')
    plt.title('Step Reward Across All Episodes (Concatenated)')
    plt.tight_layout()
    plt.show()