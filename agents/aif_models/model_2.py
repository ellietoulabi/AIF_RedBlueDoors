"""
Active Inference Generative Model (J's Version)
State factors:
    - controllable: [self_position x doors_state]
    - uncontrollable: [other_position]
Observation modalities:
    - [self_position, doors_state, near_door]
    - [other_position]
"""

from itertools import product
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pymdp import utils as pymdp_utils
from pymdp.maths import softmax as pymdp_softmax  # avoid conflict with scipy version


parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()

VERBOSE = args.verbose



"utils"

def grid_to_pos(map_size: int) -> list:
    width, height = map(int, map_size.lower().split("x"))
    pos = []
    counter = 0
    for y in range(height):
        for x in range(width):
            pos.append(f"pos_{counter}")
            counter += 1

    return pos


def my_plot_likelihood(
    A, title="", xlabel="", ylabel="", xticklabels=None, yticklabels=None
):
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


def get_next_pos_idx(pos_idx, action):
    x, y = pos_idx_to_xy[pos_idx]
    if action == 0:
        y -= 1  # up
    elif action == 1:
        y += 1  # down
    elif action == 2:
        x -= 1  # left
    elif action == 3:
        x += 1  # right
    return xy_to_pos_idx.get((x, y), pos_idx)


"Global Variables"
map_size = "3x3"
action_names = ["up", "down", "left", "right", "open"]




" States Definition "

self_pos_factor = grid_to_pos(map_size)
other_pos_factor = grid_to_pos(map_size)
doors_state_factor = [
    "red_closed_blue_closed",
    "red_open_blue_closed",
    "red_closed_blue_open",
    "red_open_blue_open",
]

controllable_state_factors = [
    f"{p}|{d}" for p, d in product(self_pos_factor, doors_state_factor)
]
num_states = [len(controllable_state_factors), len(other_pos_factor)]
num_factors = len(num_states)

if VERBOSE:
    print(f"Number of controllable states: {num_states[0]}")
    print(f"Number of uncontrollable states: {num_states[1]}")
    print(f"Number of factors: {num_factors}")


" Observation Definition "

self_pos_modality = grid_to_pos(map_size)
other_pos_modality = grid_to_pos(map_size)
door_state_modality = [
    "red_closed_blue_closed",
    "red_open_blue_closed",
    "red_closed_blue_open",
    "red_open_blue_open",
]
near_door_modality = ["not_near_door", "near_door"]  # NOTE: near_color_door?!

num_obs = [
    len(self_pos_modality),
    len(door_state_modality),
    len(near_door_modality),
    len(other_pos_modality),
]
num_modalities = len(num_obs)

num_controls = [len(action_names), 1]
num_control_factors = len(num_controls)

if VERBOSE:
    print(f"Number of self position observations: {num_obs[0]}")
    print(f"Number of door state observations: {num_obs[1]}")
    print(f"Number of near door observations: {num_obs[2]}")
    print(f"Number of other position observations: {num_obs[3]}")
    print(f"Number of modalities: {num_modalities}")
    print(f"Number of control factors: {num_control_factors}")
    print(f"Number of controls: {num_controls[0]}")


" A matrix definition "

A = pymdp_utils.initialize_empty_A(num_obs, num_states)


noise_level = 0.1  # 5% probability mass spread over incorrect observations

# self_pos_modality
num_self_pos = len(self_pos_modality)

for s0_idx, controllable_state in enumerate(controllable_state_factors):
    self_pos, _ = controllable_state.split("|")
    self_pos_idx = self_pos_modality.index(self_pos)

    for s1_idx in range(len(other_pos_factor)):
        A[0][:, s0_idx, s1_idx] = noise_level / (
            num_self_pos - 1
        )  # Spread noise across all
        A[0][self_pos_idx, s0_idx, s1_idx] = (
            1.0 - noise_level
        )  # High probability for correct obs


# door_state_modality
num_door_states = len(door_state_modality)

for s0_idx, controllable_state in enumerate(controllable_state_factors):
    _, door_state = controllable_state.split("|")  # split into self_pos and door_state
    door_state_idx = door_state_modality.index(door_state)

    for s1_idx in range(len(other_pos_factor)):
        # Initialize uniform noise over all door state observations
        A[1][:, s0_idx, s1_idx] = noise_level / (num_door_states - 1)

        # Assign high probability to the correct door state
        A[1][door_state_idx, s0_idx, s1_idx] = 1.0 - noise_level


# near_door_modality
num_near_door = len(near_door_modality)

for s0_idx in range(len(controllable_state_factors)):
    for s1_idx, other_pos_state in enumerate(other_pos_factor):

        # Start with uniform noise
        A[2][0, s0_idx, s1_idx] = noise_level / (num_near_door - 1)

        # Assign high probability to the correct other_pos observation
        A[2][1, s0_idx, s1_idx] = 1.0 - noise_level


# other_pos_modality
num_other_pos = len(other_pos_modality)

for s0_idx in range(len(controllable_state_factors)):
    for s1_idx, other_pos_state in enumerate(other_pos_factor):

        # Find the matching observation index
        other_pos_idx = other_pos_modality.index(other_pos_state)

        # Start with uniform noise
        A[3][:, s0_idx, s1_idx] = noise_level / (num_other_pos - 1)

        # Assign high probability to the correct other_pos observation
        A[3][other_pos_idx, s0_idx, s1_idx] = 1.0 - noise_level


if VERBOSE:
    for m in range(num_modalities):
        print(f"A[{m}] normalized: {pymdp_utils.is_normalized(A[m])}")

        # pymdp_utils.plot_likelihood(A[2][1,:,:])
        # print(A[0][:,:,1])


" B matrix definition " #NOTE: Does B need to be noisy?
B = pymdp_utils.initialize_empty_B(num_states, num_controls)
# for f, ns in enumerate(num_states):
#     B[f] = np.zeros((ns, ns, num_controls[f]))  # [state, state, action]


# Door transition logic


open_door_transitions = {
    "red_closed_blue_closed": [
        "red_closed_blue_closed",
        "red_open_blue_closed",
        "red_closed_blue_open",
    ],
    "red_open_blue_closed": ["red_open_blue_closed", "red_open_blue_open"],
    "red_closed_blue_open": ["red_open_blue_closed", "red_open_blue_open"],
    "red_open_blue_open": ["red_open_blue_open"],
}


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

for s_idx, combined in enumerate(controllable_state_factors):
    pos, door = combined.split("|")
    pos_idx = self_pos_factor.index(pos)
    door_idx = doors_state_factor.index(door)

    for a in range(num_controls[0]):
        if a in [0, 1, 2, 3]:  # movement
            next_pos_idx = get_next_pos_idx(pos_idx, a)
            next_state_label = f"{self_pos_factor[next_pos_idx]}|{door}"
            next_state_idx = controllable_state_factors.index(next_state_label)
            B[0][next_state_idx, s_idx, a] = 1.0

        elif a == 4:  # open
            next_doors = open_door_transitions[door]
            for next_door in next_doors:
                next_state_label = f"{pos}|{next_door}"
                next_state_idx = controllable_state_factors.index(next_state_label)
                prob = 1.0 / len(next_doors)
                B[0][next_state_idx, s_idx, a] = prob

for a_idx in range(num_controls[1]):
    B[1][:, :, a_idx] = np.eye(len(other_pos_factor))


if VERBOSE:
    for f in range(num_factors):
        print(f"B[{f}] normalizes: {pymdp_utils.is_normalized(B[f])}")

# my_plot_likelihood(
#     B[0][:, :, 1],
#     title="B[4] - Action Open",
#     xlabel="Current State",
#     ylabel="Next State",
#     xticklabels=controllable_state_factors,
#     yticklabels=controllable_state_factors,
# )


" C matrix definition "
C = pymdp_utils.obj_array_zeros(num_obs)

# 1. Self position modality
# No strong preference → uniform or zero.
C[0][:] = 0.0



# 2. Door state modality
reward_map_door_state = {
   "red_closed_blue_closed":-0.1,
    "red_open_blue_closed": 0.5,
    "red_closed_blue_open": -1.0,
    "red_open_blue_open": 1.0
}
for i, label in enumerate(door_state_modality):
    C[1][i] = reward_map_door_state[label]
    
# 3. Near door modality
near_door_preference = {
    "not_near_door": -0.1,
    "near_door": 0.3
}
for i, label in enumerate(near_door_modality):
    C[2][i] = near_door_preference[label]

# 4. Other position modality
# Neutral (no care about other agent)
C[3][:] = 0.0


temperature = 2.0

# (Optional) Normalize or softmax for numerical stability
for m in range(len(C)):
    C[m] = pymdp_softmax(C[m] / temperature)
 
if VERBOSE: 
    for m in range(num_modalities):
        print(f"C[{m}] normalized: {np.isclose(np.sum(C[0]), 1.0)}")
  
# pymdp_utils.plot_beliefs(C[1])


" D matrix definition "
"""
D matrix
"""
# Generate D matrix for prior over combined state space
D = pymdp_utils.obj_array(num_factors)

# Prior for controllable state factor
D[0] = np.zeros(num_states[0])
start_controllable_state = "pos_0|red_closed_blue_closed"
start_idx_0 = controllable_state_factors.index(start_controllable_state)
D[0][start_idx_0] = 1.0

# Prior for other agent's position (uncontrollable factor)
D[1] = np.zeros(num_states[1])
start_other_pos = "pos_0"
start_idx_1 = other_pos_factor.index(start_other_pos)
D[1][start_idx_1] = 1.0

if VERBOSE:
    print("D[0] normalized:", pymdp_utils.is_normalized(D[0]))
    print("D[1] normalized:", pymdp_utils.is_normalized(D[1]))

# pymdp_utils.plot_beliefs(D[1])


"""
pA
"""
# priors over A for learning
pA = pymdp_utils.dirichlet_like(A, scale=1.0)



MODEL = {
    "A": A,
    "B": B,
    "C": C,
    "D": D,
    "pA": pA
}



def convert_obs_to_active_inference_format(obs):
    """
    Converts raw PettingZoo observation to indices for the Active Inference modalities.
    Uses variables already defined in model_2.py (global scope).
    """
    agent0_obs = obs["agent_0"]
    agent1_obs = obs["agent_1"]
    



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
    agent_pos = tuple(agent0_obs["position"])
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

    # --- 3. Near door ---
    near_door_label = "near_door" if agent0_obs["near_door"] else "not_near_door"
    near_door_idx = near_door_modality.index(near_door_label)

    # --- 4. Other agent position ---
    other_pos = tuple(agent1_obs["position"])
    other_pos_label = xy_to_pos_label.get(other_pos, "pos_0")
    other_pos_idx = other_pos_modality.index(other_pos_label)

    return [self_pos_idx, door_state_idx, near_door_idx, other_pos_idx]






if VERBOSE:
    print("Model Configured successfully.")