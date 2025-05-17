def get_config_path(config_paths, episode, k, alternate=False):
    if alternate:
        return config_paths[(episode // k) % 2]  # Alternate between config.json and config2.json
    return config_paths[0]  # Cycle through the config paths
