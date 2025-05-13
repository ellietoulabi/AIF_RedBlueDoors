def convert_obs_to_active_inference_format(obs):
    """
    Converts raw PettingZoo observation to indices for the Active Inference modalities.
    Uses variables already defined in model_2.py (global scope).
    """
    agent_obs = obs["agent_0"]
    other_obs = obs["agent_1"]

    # --- 1. Self position ---
    agent_pos = tuple(agent_obs["position"])
    self_pos_label = xy_to_pos_label.get(agent_pos, "pos_0")
    self_pos_idx = self_pos_modality.index(self_pos_label)

    # --- 2. Door state ---
    red = agent_obs["red_door_opened"]
    blue = agent_obs["blue_door_opened"]
    if red and blue:
        door_state_label = "red_open_blue_open"
    elif red:
        door_state_label = "red_open_blue_closed"
    elif blue:
        door_state_label = "red_closed_blue_open"
    else:
        door_state_label = "red_closed_blue_closed"
    door_state_idx = door_state_modality.index(door_state_label)

    # --- 3. Near door ---
    near_door_label = "near_door" if agent_obs["near_door"] else "not_near_door"
    near_door_idx = near_door_modality.index(near_door_label)

    # --- 4. Other agent position ---
    other_pos = tuple(other_obs["position"])
    other_pos_label = xy_to_pos_label.get(other_pos, "pos_0")
    other_pos_idx = other_pos_modality.index(other_pos_label)

    return [self_pos_idx, door_state_idx, near_door_idx, other_pos_idx]
