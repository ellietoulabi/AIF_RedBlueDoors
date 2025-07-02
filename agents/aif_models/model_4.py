"Imports"

from pymdp import utils as pymdp_utils
from pymdp.maths import softmax as pymdp_softmax

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"Utills"


def my_plot_likelihood(
    A, title="", xlabel="", ylabel="", xticklabels=None, yticklabels=None
) -> None:
    """
    Plots a 2D likelihood heatmap.

    Args:
        A (np.ndarray): Matrix to visualize.
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        xticklabels (list): Labels for columns (states).
        yticklabels (list): Labels for rows (observations).
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        A, cmap="OrRd", linewidth=1.5, xticklabels=xticklabels, yticklabels=yticklabels
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


def grid_to_pos(map_size: str) -> list:
    """
    Converts a map size to a list of positions.

    Args:
        map_size (str): The size of the map in the format "widthxheight".

    Returns:
        list: A list of positions in the format "pos_0", "pos_1", etc.
    """
    width, height = map(int, map_size.lower().split("x"))
    pos = []
    counter = 0
    for y in range(height):
        for x in range(width):
            pos.append(f"pos_{counter}")
            counter += 1

    return pos


def convert_obs_to_active_inference_format(obs, agent_id):
    """
    Converts raw PettingZoo observation to indices for the Active Inference modalities.
    Uses variables already defined in model_2.py (global scope).
    """
    agents = ["agent_0", "agent_1"] 
    
    self_agent_obs = obs[agent_id]
    other_agent_obs= obs[next(a for a in agents if a != agent_id)]
    

    xy_to_pos_label = {
            (1, 1): "pos_0",
            (2, 1): "pos_1",
            (3, 1): "pos_2",
            (1, 2): "pos_3",
            (2, 2): "pos_4",
            (3, 2): "pos_5",
            (1, 3): "pos_6",
            (2, 3): "pos_7",
            (3, 3): "pos_8",
        }




    # --- 1. Self position ---
    agent_pos = tuple(self_agent_obs["position"])
    self_pos_label = xy_to_pos_label.get(agent_pos, "pos_0")
    self_pos_idx = self_pos_modality.index(self_pos_label)

    # --- 2. Door state ---
    red = obs["red_door_opened"]
    blue = obs["blue_door_opened"]
    if red and blue:
        door_state_label = "red_open_blue_open"
    elif red:
        door_state_label = "red_open_blue_closed"
    elif blue:
        door_state_label = "red_closed_blue_open"
    else:
        door_state_label = "red_closed_blue_closed"
    door_state_idx = door_state_modality.index(door_state_label)

    # --- 3. Other intention ---
    if other_agent_obs["near_red_door"]:
        other_intention_label = "open_red_next"
    elif other_agent_obs["near_blue_door"]:
        other_intention_label = "open_blue_next"
    else:
        other_intention_label = "idle"
    other_intention_idx = other_intention_modality.index(other_intention_label)




    
    # --- 4. Other agent position ---
    other_pos = tuple(other_agent_obs["position"])
    other_pos_label = xy_to_pos_label.get(other_pos, "pos_0")
    other_pos_idx = other_pos_modality.index(other_pos_label)

    return [self_pos_idx, other_pos_idx, door_state_idx, other_intention_idx]





"Global Variables"

MAP_SIZE = "3x3"
VERBOSE = False
A_NOISE_LEVEL = 0.2
B_NOISE_LEVEL = 0.2


"States Definition"

self_pos_factor = grid_to_pos(MAP_SIZE)
other_pos_factor = grid_to_pos(MAP_SIZE)
doors_state_factor = [
    "red_closed_blue_closed",
    "red_open_blue_closed",
    "red_closed_blue_open",
    "red_open_blue_open",
]
other_intention_factor = [
    "open_red_next",
    "open_blue_next",
    "idle",
]

num_states = [
    len(self_pos_factor),
    len(other_pos_factor),
    len(doors_state_factor),
    len(other_intention_factor),
]
num_factors = len(num_states)

if VERBOSE:
    print("\n", "-" * 10 + "States Definition" + "-" * 10)
    print(f"Number of factors: {num_factors}")
    print(f"Number of states in self position factor: {num_states[0]}")
    print(f"Number of states in other position factor: {num_states[1]}")
    print(f"Number of states in doors state factor: {num_states[2]}")
    print(f"Number of states in other intention factor: {num_states[3]}")


"Observation Definition"

self_pos_modality = grid_to_pos(MAP_SIZE)
other_pos_modality = grid_to_pos(MAP_SIZE)
door_state_modality = [
    "red_closed_blue_closed",
    "red_open_blue_closed",
    "red_closed_blue_open",
    "red_open_blue_open",
]
other_intention_modality = [
    "open_red_next",
    "open_blue_next",
    "idle",
]
game_outcome_modality = ["win", "lose", "neutral"]

num_obs = [
    len(self_pos_modality),
    len(other_pos_modality),
    len(door_state_modality),
    len(other_intention_modality),
    len(game_outcome_modality),
]
num_modalities = len(num_obs)

if VERBOSE:
    print("\n", "-" * 10 + "Observation Definition" + "-" * 10)
    print(f"Number of modalities: {num_modalities}")
    print(f"Number of self position observations: {num_obs[0]}")
    print(f"Number of other position observations: {num_obs[1]}")
    print(f"Number of door state observations: {num_obs[2]}")
    print(f"Number of other intention observations: {num_obs[3]}")
    print(f"Number of game outcome observations: {num_obs[4]}")


"Control Definition"

self_movement_action_names = ["up", "down", "left", "right"]
door_action_names = ["noop", "open"]
other_intention_action_names = ["noop"]
other_movement_action_names = ["noop"]

num_controls = [
    len(self_movement_action_names),
    len(other_movement_action_names),
    len(door_action_names),
    len(other_intention_action_names),
]
num_control_factors = len(num_controls)

if VERBOSE:
    print("\n", "-" * 10 + "Control Definition" + "-" * 10)
    print(f"Number of control factors: {num_control_factors}")
    print(f"Number of self movement controls: {num_controls[0]}")
    print(f"Number of other movement controls: {num_controls[1]}")
    print(f"Number of door controls: {num_controls[2]}")
    print(f"Number of other intention controls: {num_controls[3]}")


"A Matrix Definition"

"""
A (likelihood matrix) is probability of seeing an observation given a state p(o|s).
A[i] is the likelihood matrix for the i-th modality.
A[i] shape is [num_obs[i], num_states[0], ..., num_states[N]]
"""


A = pymdp_utils.initialize_empty_A(num_obs, num_states)


# --- Modality 0: Self‐Position (noisy identity on s0) ---
for s0 in range(num_states[0]):
    for s1 in range(num_states[1]):
        for s2 in range(num_states[2]):
            for s3 in range(num_states[3]):
                for o_idx in range(num_obs[0]):
                    if o_idx == s0:
                        A[0][o_idx, s0, :, :, :] = 1.0 - A_NOISE_LEVEL
                    else:
                        A[0][o_idx, s0, :, :, :] = A_NOISE_LEVEL / (num_obs[0] - 1)


# --- Modality 1: Other‐Position (noisy identity on s1) ---
for s0 in range(num_states[0]):
    for s1 in range(num_states[1]):
        for s2 in range(num_states[2]):
            for s3 in range(num_states[3]):
                for o_idx in range(num_obs[1]):
                    if o_idx == s1:
                        A[1][o_idx, :, s1, :, :] = 1.0 - A_NOISE_LEVEL
                    else:
                        A[1][o_idx, :, s1, :, :] = A_NOISE_LEVEL / (num_obs[1] - 1)


# --- Modality 2: Door State (noisy identity on s2) ---
for s0 in range(num_states[0]):
    for s1 in range(num_states[1]):
        for s2 in range(num_states[2]):
            for s3 in range(num_states[3]):
                for o_idx in range(num_obs[2]):
                    if o_idx == s2:
                        A[2][o_idx, :, :, s2, :] = 1.0 - A_NOISE_LEVEL
                    else:
                        A[2][o_idx, :, :, s2, :] = A_NOISE_LEVEL / (num_obs[2] - 1)


# --- Modality 3: Other Intention (noisy identity on s3) ---
for s0 in range(num_states[0]):
    for s1 in range(num_states[1]):
        for s2 in range(num_states[2]):
            for s3 in range(num_states[3]):
                for o_idx in range(num_obs[3]):
                    if o_idx == s3:
                        A[3][o_idx, :, :, :, s3] = 1.0 - A_NOISE_LEVEL
                    else:
                        A[3][o_idx, :, :, :, s3] = A_NOISE_LEVEL / (num_obs[3] - 1)


# --- Modality 4: Game Outcome (noisy identity on s4) ---
neutral_idx = game_outcome_modality.index("neutral")
win_idx = game_outcome_modality.index("win")
lose_idx = game_outcome_modality.index("lose")
terminal_idx = doors_state_factor.index("red_open_blue_open")

for s0 in range(num_states[0]):  # self_pos
    for s1 in range(num_states[1]):  # other_pos
        for s2 in range(num_states[2]):  # doors_state
            for s3 in range(num_states[3]):  # other_intent
                # Win
                A[4][0, s0, s1, 3, s3] = 1.0 - A_NOISE_LEVEL
                A[4][0, s0, s1, 0:3, s3] = A_NOISE_LEVEL / (num_obs[4] - 1)
                # Lose
                A[4][1, s0, s1, 2, s3] = 1.0 - A_NOISE_LEVEL
                A[4][1, s0, s1, 0:2, s3] = A_NOISE_LEVEL / (num_obs[4] - 1)
                A[4][1, s0, s1, 3, s3] = A_NOISE_LEVEL / (num_obs[4] - 1)
                # Neutral
                A[4][2, s0, s1, 0, s3] = 1.0 - A_NOISE_LEVEL
                A[4][2, s0, s1, 1, s3] = 1.0 - A_NOISE_LEVEL
                A[4][2, s0, s1, 2:4, s3] = A_NOISE_LEVEL / (num_obs[4] - 1)

if VERBOSE:
    for m in range(num_modalities):
        print(f"A[{m}] normalized: {pymdp_utils.is_normalized(A[m])}")


" B Matrix Definition "

B = pymdp_utils.initialize_empty_B(num_states, num_controls)


# Initialize B matrices with zeros
for f in range(num_factors):
    B[f].fill(0.0)


# Helper function to get next position index
def get_next_pos_idx(pos_idx, action):
    """Get next position index based on current position and action"""
    # 3x3 grid mapping
    pos_idx_to_xy = {
        0: (0, 0),
        1: (1, 0),
        2: (2, 0),
        3: (0, 1),
        4: (1, 1),
        5: (2, 1),
        6: (0, 2),
        7: (1, 2),
        8: (2, 2),
    }
    xy_to_pos_idx = {v: k for k, v in pos_idx_to_xy.items()}

    x, y = pos_idx_to_xy[pos_idx]
    if action == 0:  # up
        y = max(0, y - 1)
    elif action == 1:  # down
        y = min(2, y + 1)
    elif action == 2:  # left
        x = max(0, x - 1)
    elif action == 3:  # right
        x = min(2, x + 1)

    return xy_to_pos_idx.get((x, y), pos_idx)


# B[0]: Self position transitions (controlled by self movement actions)
for current_pos in range(num_states[0]):
    for action in range(num_controls[0]):
        if action < 4:  # Movement actions (up, down, left, right)
            next_pos = get_next_pos_idx(current_pos, action)
            # Add noise to movement transitions
            B[0][next_pos, current_pos, action] = 1.0 - B_NOISE_LEVEL
            # Distribute noise to other possible positions
            for other_pos in range(num_states[0]):
                if other_pos != next_pos:
                    B[0][other_pos, current_pos, action] += B_NOISE_LEVEL / (
                        num_states[0] - 1
                    )
        else:  # No movement action
            B[0][current_pos, current_pos, action] = 1.0 - B_NOISE_LEVEL
            # Distribute noise to other positions
            for other_pos in range(num_states[0]):
                if other_pos != current_pos:
                    B[0][other_pos, current_pos, action] += B_NOISE_LEVEL / (
                        num_states[0] - 1
                    )

# B[1]: Other position transitions (controlled by other movement actions - mostly noop)
for current_pos in range(num_states[1]):
    for action in range(num_controls[1]):
        # Other agent mostly stays in place (noop action) with noise
        B[1][current_pos, current_pos, action] = 1.0 - B_NOISE_LEVEL
        # Distribute noise to other positions
        for other_pos in range(num_states[1]):
            if other_pos != current_pos:
                B[1][other_pos, current_pos, action] += B_NOISE_LEVEL / (
                    num_states[1] - 1
                )

# B[2]: Door state transitions (controlled by door actions)
door_transitions = {
    0: [0, 1, 2],  # red_closed_blue_closed -> can open red or blue
    1: [1, 3],  # red_open_blue_closed -> can open blue
    2: [2, 3],  # red_closed_blue_open -> can open red
    3: [3],  # red_open_blue_open -> terminal state
}

for current_door_state in range(num_states[2]):
    for action in range(num_controls[2]):
        if action == 0:  # noop
            # Stay in current state with noise
            B[2][current_door_state, current_door_state, action] = 1.0 - B_NOISE_LEVEL
            # Distribute noise to other door states
            for other_state in range(num_states[2]):
                if other_state != current_door_state:
                    B[2][other_state, current_door_state, action] += B_NOISE_LEVEL / (
                        num_states[2] - 1
                    )
        elif action == 1:  # open
            possible_next_states = door_transitions[current_door_state]
            prob = (1.0 - B_NOISE_LEVEL) / len(possible_next_states)
            for next_state in possible_next_states:
                B[2][next_state, current_door_state, action] = prob
            # Distribute noise to other door states
            for other_state in range(num_states[2]):
                if other_state not in possible_next_states:
                    B[2][other_state, current_door_state, action] += B_NOISE_LEVEL / (
                        num_states[2] - len(possible_next_states)
                    )

# B[3]: Other intention transitions (controlled by other intention actions - mostly noop)
for current_intent in range(num_states[3]):
    for action in range(num_controls[3]):
        # Other agent's intention mostly stays the same (noop action) with noise
        B[3][current_intent, current_intent, action] = 1.0 - B_NOISE_LEVEL
        # Distribute noise to other intentions
        for other_intent in range(num_states[3]):
            if other_intent != current_intent:
                B[3][other_intent, current_intent, action] += B_NOISE_LEVEL / (
                    num_states[3] - 1
                )

# Verify B matrices are normalized
if VERBOSE:
    print("\n", "-" * 10 + "B Matrix Definition " + "-" * 10)
    print(f"B matrix shape: {[B[i].shape for i in range(len(B))]}")
    for i, shape in enumerate([B[i].shape for i in range(len(B))]):
        print(f"B[{i}] shape: {shape}")
    for f in range(num_factors):
        print(f"B[{f}] normalized: {pymdp_utils.is_normalized(B[f])}")

    # B[0]: Self position transitions
    for action_idx in range(num_controls[0]):
        action_name = (
            self_movement_action_names[action_idx]
            if action_idx < len(self_movement_action_names)
            else f"action_{action_idx}"
        )
        my_plot_likelihood(
            B[0][:, :, action_idx],
            title=f"B[0] - Self Position Transitions ({action_name})",
            xlabel="Current Position",
            ylabel="Next Position",
            xticklabels=self_pos_factor,
            yticklabels=self_pos_factor,
        )

    # B[1]: Other position transitions
    for action_idx in range(num_controls[1]):
        action_name = (
            other_movement_action_names[action_idx]
            if action_idx < len(other_movement_action_names)
            else f"action_{action_idx}"
        )
        my_plot_likelihood(
            B[1][:, :, action_idx],
            title=f"B[1] - Other Position Transitions ({action_name})",
            xlabel="Current Position",
            ylabel="Next Position",
            xticklabels=other_pos_factor,
            yticklabels=other_pos_factor,
        )

    # B[2]: Door state transitions
    for action_idx in range(num_controls[2]):
        action_name = (
            door_action_names[action_idx]
            if action_idx < len(door_action_names)
            else f"action_{action_idx}"
        )
        my_plot_likelihood(
            B[2][:, :, action_idx],
            title=f"B[2] - Door State Transitions ({action_name})",
            xlabel="Current Door State",
            ylabel="Next Door State",
            xticklabels=doors_state_factor,
            yticklabels=doors_state_factor,
        )

    # B[3]: Other intention transitions
    for action_idx in range(num_controls[3]):
        action_name = (
            other_intention_action_names[action_idx]
            if action_idx < len(other_intention_action_names)
            else f"action_{action_idx}"
        )
        my_plot_likelihood(
            B[3][:, :, action_idx],
            title=f"B[3] - Other Intention Transitions ({action_name})",
            xlabel="Current Intention",
            ylabel="Next Intention",
            xticklabels=other_intention_factor,
            yticklabels=other_intention_factor,
        )



" C Matrix Definition  "
C = pymdp_utils.obj_array_zeros(num_obs)


num_obs = [
    len(self_pos_modality),
    len(other_pos_modality),
    len(door_state_modality),
    len(other_intention_modality),
    len(game_outcome_modality),
]


# Self position, other position, other intention modalities
# No strong preference → uniform or zero.
C[0][:] = 0.0
C[1][:] = 0.0
C[3][:] = 0.0

# Door state modality
reward_map_door_state = {
   "red_closed_blue_closed":-0.01,
    "red_open_blue_closed": 2.5,
    "red_closed_blue_open": -5.0,
    "red_open_blue_open": 5.0
}
for i, label in enumerate(door_state_modality):
    C[2][i] = reward_map_door_state[label]
    
# Game outcome modality
C[4][:] = [+5.0, -5.0, 0.0]


temperature = 2.0
for m in range(len(C)):
    C[m] = pymdp_softmax(C[m] / temperature)
 
 
if VERBOSE: 
    print("\n", "-" * 10 + "C Matrix Definition " + "-" * 10)
    print(f"C shape: {[C[m].shape for m in range(len(C))]}")
    for m in range(num_modalities):
        print(f"C[{m}] normalized: {np.isclose(np.sum(C[0]), 1.0)}")

# pymdp_utils.plot_beliefs(C[4])



" D Matrix Definition "
D1 = pymdp_utils.obj_array(num_factors)
D1[0] = np.zeros(num_states[0])
D1[1] = np.zeros(num_states[1])
D1[2] = np.zeros(num_states[2])
D1[3] = np.zeros(num_states[3])

D1[0][0] = 1.0
D1[1][8] = 1.0
D1[2][0] = 1.0
D1[3][:] = 1.0 / len(D1[3])




if VERBOSE:
    print("\n", "-" * 10 + "D Matrix Definition " + "-" * 10)
    for f in range(len(D1)):
        print(f"D[{f}] sum: {np.sum(D1[f])}, normalized: {np.isclose(np.sum(D1[f]), 1.0)}")


"pA Matrix Definition"
pA = pymdp_utils.dirichlet_like(A, scale=1.0)




MODEL = {
    "A": A,
    "B": B,
    "C": C,
    "D": D1,
    "pA": pA
}

