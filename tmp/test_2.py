import random
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt

from pymdp.agent import Agent

sys.path.append("..")
from envs.redbluedoors_env.ma_redbluedoors import RedBlueDoorEnv
from agents.aif_models import model_2
from agents.aif_models.model_2 import convert_obs_to_active_inference_format

if __name__ == "__main__":
    
    seed = 42
    max_steps = 10
    np.random.seed(seed)
    random.seed(seed)
   
    aif_agent = Agent(
        A=model_2.MODEL["A"],
        B=model_2.MODEL["B"],
        C=model_2.MODEL["C"],
        D=model_2.MODEL["D"],
        pA=model_2.MODEL["pA"],
        inference_algo="MMP",
        policy_len=2,
        inference_horizon=1,
        sampling_mode="marginal",
        action_selection="stochastic",
        alpha=0.1,
    )
    print(aif_agent.control_fac_idx)
    
    print(np.array(aif_agent.policies).shape)
   
    config_path = "../envs/redbluedoors_env/configs/config.json"
    env = RedBlueDoorEnv(max_steps=max_steps, config_path=config_path)
    obs, _ = env.reset()
    print(obs)
    print(obs["agent_0"]["next_intention"])
    exit()
    
    aif_obs = convert_obs_to_active_inference_format(obs, "agent_0")
    total_reward = 0
    step_rewards = []
    aif_agent.D = copy.deepcopy(model_2.MODEL["D"])

    for step in range(max_steps):
        qs = aif_agent.infer_states(aif_obs)
        aif_agent.D = qs
        q_pi, G = aif_agent.infer_policies()
        action = aif_agent.sample_action()
        action_int = int(action[0])
        action_dict = {"agent_0": action_int}
        obs, rewards, terminations, truncations, infos = env.step(action_dict)
        print(infos["agent_0"]["action_meaning"],"\n",infos["map"])
        reward = rewards.get("agent_0", 0)
        total_reward += reward
        step_rewards.append(reward)
        if any(terminations.values()) or any(truncations.values()):
            break
        aif_obs = convert_obs_to_active_inference_format(obs, "agent_0")
    env.close()

    print(f"Total reward: {total_reward}")
    plt.figure(figsize=(10, 4))
    plt.plot(step_rewards, label='Step Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Step Reward for Single Episode')
    plt.tight_layout()
    plt.show()