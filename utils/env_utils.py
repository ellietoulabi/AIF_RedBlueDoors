def get_config_path(config_paths, episode, k):
    return config_paths[(episode // k) % 2]  # Alternate between config.json and config2.json
