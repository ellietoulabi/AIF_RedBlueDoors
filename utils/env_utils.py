def get_config_path(config_paths, episode):
    return config_paths[episode % 2]  # Alternate between config.json and config2.json
