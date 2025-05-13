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
            "aif_reward"
        ]
        writer = csv.writer(file)
        writer.writerow(fieldnames)

    EPISODES = episodes
    MAX_STEPS = max_steps

    config_paths = [
        "envs/redbluedoors_env/configs/config.json",
        "envs/redbluedoors_env/configs/config2.json",
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
                ql_agent.update_q_table(state, action_dict, rewards, next_state, "agent_1")

            total_reward_aif += rewards.get("agent_0", 0)
            total_reward_ql += rewards.get("agent_1", 0)

            state = next_state

            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    seed,
                    episode,
                    step,
                    int(next_action_ql),
                    int(next_action_aif[0]),
                    rewards.get("agent_1", 0),
                    rewards.get("agent_0", 0)
                ])

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
    q_table_file = f"q_table_seed_{seed}.json"
    log_file = f"log_seed_{seed}.csv"
    rewards_aif, rewards_ql = run_experiment(seed, q_table_file, log_file)

    for ep, (ra, rq) in enumerate(zip(rewards_aif, rewards_ql)):
        all_results.append({"seed": seed, "episode": ep, "aif_reward": ra, "ql_reward": rq})
