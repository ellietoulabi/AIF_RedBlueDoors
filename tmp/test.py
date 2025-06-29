import os
import csv
import random
import numpy as np
import copy
from tqdm import trange
import sys
import matplotlib.pyplot as plt

from pymdp.agent import Agent

sys.path.append("..")
from envs.redbluedoors_env.ma_redbluedoors import RedBlueDoorEnv
from agents.aif_models import model_2
from agents.aif_models.model_2 import convert_obs_to_active_inference_format
from utils.env_utils import get_config_path
from utils.logging_utils import create_experiment_folder


log_paths = create_experiment_folder(base_dir=".")
print("Logging folders:")
for k, v in log_paths.items():
    print(f"{k}: {v}")

# Main script - simple single agent AIF loop
if __name__ == "__main__":
    # Parameters
    seed = 42
    episodes = 5
    max_steps = 100
    
    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)

    print(f"Running single agent AIF with seed {seed}")
    
    # Create AIF agent
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
        config_path = ig_paths =  "../envs/redbluedoors_env/configs/config.json"
        env = RedBlueDoorEnv(max_steps=max_steps, config_path=config_path)
        
        # Reset environment
        obs, _ = env.reset()
        aif_obs = convert_obs_to_active_inference_format(obs,"agent_0")
        total_reward = 0
        step_rewards = []  # Store rewards for this episode
        
        # Reset agent's initial beliefs
        aif_agent.D = copy.deepcopy(model_2.MODEL["D"])

        for step in range(max_steps):
            # AIF inference loop
            qs = aif_agent.infer_states(aif_obs)
            aif_agent.D = qs
            q_pi, G = aif_agent.infer_policies()

            # Sample action
            action = aif_agent.sample_action()
            action_int = int(action[0])
            
            # Take action (single agent)
            action_dict = {"agent_0": action_int}
            obs, rewards, terminations, truncations, infos = env.step(action_dict)

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