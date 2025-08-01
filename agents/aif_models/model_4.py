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
    elif action == 4:  # noop
        pass  # Stay in the same position

    return xy_to_pos_idx.get((x, y), pos_idx)


    
 






"Global Variables"

MAP_SIZE = "3x3"
VERBOSE = False
A_NOISE_LEVEL = 0.2
B_NOISE_LEVEL = 0.2



# AGENT_0_POLICIES = [
# # Goal-directed
# np.array([
#     [1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0],
#     [4, 0, 0, 0, 0],
#     [4, 0, 0, 0, 0],
# ]),
# np.array([
#     [3, 0, 0, 0, 0],
#     [3, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0],
#     [4, 0, 0, 0, 0],
#     [4, 0, 0, 0, 0],
# ]),
# # Delayed goal-directed
# np.array([
#     [4, 0, 0, 0, 0],
#     [4, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0],
# ]),
# np.array([
#     [4, 0, 0, 0, 0],
#     [4, 0, 0, 0, 0],
#     [3, 0, 0, 0, 0],
#     [3, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0],
# ]),
# # Switched 
# np.array([
#     [3, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [2, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0],
# ]),
# np.array([
#     [1, 0, 0, 0, 0],
#     [3, 0, 0, 0, 0],
#     [3, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0],
#     [4, 0, 0, 0, 0],
# ]),
# # Ineffective 
# np.array([
#     [3, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0],
# ]),
# np.array([
#     [3, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [2, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0],
# ]),  
       
# ]

# AGENT_1_POLICIES = [   
# # Goal-directed
# np.array([
#     [2, 0, 0, 0, 0, 0],
#     [2, 0, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0, 0],
#     [4, 0, 0, 0, 0, 0],
#     [4, 0, 0, 0, 0, 0],
# ]),
# np.array([
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0, 0],
#     [4, 0, 0, 0, 0, 0],
#     [4, 0, 0, 0, 0, 0],
# ]),
# # Delayed goal-directed
# np.array([
#     [4, 0, 0, 0, 0, 0],
#     [4, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0],
# ]),
# np.array([
#     [4, 0, 0, 0, 0, 0],
#     [4, 0, 0, 0, 0, 0],
#     [2, 0, 0, 0, 0, 0],
#     [2, 0, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0, 0],
# ]),
# # Switched goal-directed
# np.array([
#     [0, 0, 0, 0, 0, 0],
#     [2, 0, 0, 0, 0, 0],
#     [2, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0, 0],
# ]),
# np.array([
#     [2, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [3, 0, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0, 0],
# ]),
# # Ineffective
# np.array([
#     [2, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0, 0],
# ]),
# np.array([
#     [0, 0, 0, 0, 0, 0],
#     [2, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
#     [3, 0, 0, 0, 0, 0],
#     [4, 0, 1, 0, 0, 0],
# ]),  
  
# ]


AGENT_0_POLICIES = [
# Goal-directed
np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [4, 0, 0, 0],
    [4, 0, 0, 0],
    [4, 0, 0, 0],
]),
np.array([
    [3, 0, 0, 0],
    [3, 0, 0, 0],
    [4, 0, 0, 0],
    [4, 0, 0, 0],
    [4, 0, 0, 0],
]),
# Delayed goal-directed
np.array([
    [4, 0, 0, 0],
    [4, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [4, 0, 0, 0],
]),
np.array([
    [4, 0, 0, 0],
    [4, 0, 0, 0],
    [3, 0, 0, 0],
    [3, 0, 0, 0],
    [4, 0, 0, 0],
]),
# Switched 
np.array([
    [3, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [2, 0, 0, 0],
    [4, 0, 0, 0],
]),
np.array([
    [1, 0, 0, 0],
    [3, 0, 0, 0],
    [3, 0, 0, 0],  
    [0, 0, 0, 0],
    [4, 0, 0, 0],
]),
# Ineffective 
np.array([
    [3, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [4, 0, 0, 0],
]),
np.array([
    [3, 0, 0, 0],
    [1, 0, 0, 0],
    [2, 0, 0, 0],
    [0, 0, 0, 0],
    [4, 0, 0, 0],
]),  
       
]



"States Definition"

self_pos_factor = grid_to_pos(MAP_SIZE)
other_pos_factor = grid_to_pos(MAP_SIZE)
red_door_pos_factor = grid_to_pos(MAP_SIZE)
blue_door_pos_factor = grid_to_pos(MAP_SIZE)
num_states = [
    len(self_pos_factor),
    len(other_pos_factor),
    len(red_door_pos_factor),
    len(blue_door_pos_factor),
]
num_factors = len(num_states)


self_pos_modality = grid_to_pos(MAP_SIZE)
other_pos_modality = grid_to_pos(MAP_SIZE)
near_red_door_modality = ["near", "far"]
near_blue_door_modality = ["near", "far"]
num_obs = [
    len(self_pos_modality),
    len(other_pos_modality),
    len(near_red_door_modality),
    len(near_blue_door_modality),
]
num_modalities = len(num_obs)


self_pos_controls = ["up", "down", "left", "right", "noop"]
num_controls = [
    len(self_pos_controls),
    1,
    1,
    1,
]
num_control_factors = len(num_controls)


A = pymdp_utils.initialize_empty_A(num_obs, num_states)
base0 = np.full((num_obs[0], num_states[0]), A_NOISE_LEVEL/(num_obs[0]-1))   
np.fill_diagonal(base0, 1.0 - A_NOISE_LEVEL)
A[0][:] = base0.reshape(num_obs[0], num_states[0], *([1] * (A[0].ndim - 2)))
base1 = np.full((num_obs[1], num_states[1]), A_NOISE_LEVEL / (num_obs[1] - 1))
np.fill_diagonal(base1, 1.0 - A_NOISE_LEVEL)
A[1][:] = base1.reshape([num_obs[1], 1, num_states[1]] + [1] * (A[1].ndim - 3))
A[2].fill(1 / num_obs[2])
A[3].fill(1 / num_obs[3])


B = pymdp_utils.initialize_empty_B(num_states, num_controls)
for f in range(num_factors):
    B[f].fill(0.0)
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


S2 = num_states[2]
baseB2 = np.full((S2, S2), B_NOISE_LEVEL/(S2-1))
np.fill_diagonal(baseB2, 1.0 - B_NOISE_LEVEL)
B[2][:] = baseB2.reshape(S2, S2, 1)

S3 = num_states[3]
baseB3 = np.full((S3, S3), B_NOISE_LEVEL/(S3-1))
np.fill_diagonal(baseB3, 1.0 - B_NOISE_LEVEL)
B[3][:] = baseB3.reshape(S3, S3, 1)



C = pymdp_utils.obj_array_zeros(num_obs)
C[0][:] = 0.0
C[1][:] = 0.0


# near red door modality
reward_map_near_door = {
    "near": 10.0,
    "far": -0.1,
}   
for i, label in enumerate(near_red_door_modality):
    C[2][i] = reward_map_near_door[label]

for i, label in enumerate(near_blue_door_modality):
    C[3][i] = reward_map_near_door[label]

# temperature = 2.0
# for m in range(len(C)):
#     C[m] = pymdp_softmax(C[m] / temperature)
    

D1 = pymdp_utils.obj_array(num_factors)
D1[0] = np.zeros(num_states[0])
D1[1] = np.zeros(num_states[1])
D1[2] = np.zeros(num_states[2])
D1[3] = np.zeros(num_states[3])


D1[0][0] = 1.0
D1[1][8] = 1.0
D1[2][:] = 1.0 / len(D1[2])
D1[3][:] = 1.0 / len(D1[3])


"pA Matrix Definition"
pA = pymdp_utils.dirichlet_like(A, scale=1.0)


MODEL = {
    "A": A,
    "B": B,
    "C": C,
    "D": D1,
    "pA": pA,
    "Policies":AGENT_0_POLICIES
}






def convert_obs_to_active_inference_format(obs, agent_id)->list:

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
    
     # --- 2. Other agent position ---
    other_pos = tuple(other_agent_obs["position"])
    other_pos_label = xy_to_pos_label.get(other_pos, "pos_0")
    other_pos_idx = other_pos_modality.index(other_pos_label)

    # --- 3. Near red door ---
    near_red_door = self_agent_obs["near_red_door"]
    if near_red_door:
        near_red_door_label = "near"
    else:
        near_red_door_label = "far"
    near_red_door_idx = near_red_door_modality.index(near_red_door_label)

    
    # --- 4. Near blue door ---
    near_blue_door = self_agent_obs["near_blue_door"]
    if near_blue_door:
        near_blue_door_label = "near"
    else:
        near_blue_door_label = "far"
    near_blue_door_idx = near_blue_door_modality.index(near_blue_door_label)
    

    return [self_pos_idx, other_pos_idx, near_red_door_idx, near_blue_door_idx]






# self_pos_factor = grid_to_pos(MAP_SIZE)
# other_pos_factor = grid_to_pos(MAP_SIZE)
# doors_state_factor = [
#     "red_closed_blue_closed",
#     "red_open_blue_closed",
#     "red_closed_blue_open",
#     "red_open_blue_open",
# ]
# red_door_pos_factor = grid_to_pos(MAP_SIZE)
# blue_door_pos_factor = grid_to_pos(MAP_SIZE)
# # joint_policy_factor = [ (i,j) for i in range(len(AGENT_0_POLICIES)) for j in range(len(AGENT_1_POLICIES)) ]


# num_states = [
#     len(self_pos_factor),
#     len(other_pos_factor),
#     len(doors_state_factor),
#     len(red_door_pos_factor),
#     len(blue_door_pos_factor),
#     # len(joint_policy_factor),
# ]
# num_factors = len(num_states)

# if VERBOSE:
#     print("\n", "-" * 10 + "States Definition" + "-" * 10)
#     print(f"Number of factors: {num_factors}")
#     print(f"Number of states in self position factor: {num_states[0]}")
#     print(f"Number of states in other position factor: {num_states[1]}")
#     print(f"Number of states in doors state factor: {num_states[2]}")
#     print(f"Number of states in red door position factor: {num_states[3]}")
#     print(f"Number of states in blue door position factor: {num_states[4]}")
#     # print(f"Number of states in joint policy factor: {num_states[5]}")



# "Observation Definition"

# self_pos_modality = grid_to_pos(MAP_SIZE)
# other_pos_modality = grid_to_pos(MAP_SIZE)
# door_state_modality = [
#     "red_closed_blue_closed",
#     "red_open_blue_closed",
#     "red_closed_blue_open",
#     "red_open_blue_open",
# ]
# near_red_door_modality = [
#     "near",
#     "far",
# ]
# near_blue_door_modality = [
#     "near",
#     "far",
# ]

# # ALL_ACTIONS = [
# #     "up",
# #     "down",
# #     "left",
# #     "right",
# #     "noop",
# #     "door_noop",
# #     "door_open",
# # ]
# # joint_action_modality = [
# #     (i,j) for i in range(len(ALL_ACTIONS)) for j in range(len(ALL_ACTIONS))
# # ]
# # game_outcome_modality = ["win", "lose", "neutral"]

# num_obs = [
#     len(self_pos_modality),
#     len(other_pos_modality),
#     len(door_state_modality),
#     len(near_red_door_modality),
#     len(near_blue_door_modality),
#     # len(joint_action_modality),
# ]
# num_modalities = len(num_obs)

# if VERBOSE:
#     print("\n", "-" * 10 + "Observation Definition" + "-" * 10)
#     print(f"Number of modalities: {num_modalities}")
#     print(f"Number of self position observations: {num_obs[0]}")
#     print(f"Number of other position observations: {num_obs[1]}")
#     print(f"Number of door state observations: {num_obs[2]}")
#     print(f"Number of near red door observations: {num_obs[3]}")
#     print(f"Number of near blue door observations: {num_obs[4]}")
#     # print(f"Number of joint action observations: {num_obs[5]}")


# "Control Definition"

# self_pos_controls = ["up", "down", "left", "right", "noop"]
# other_pos_controls = ["noop"]
# doors_state_controls = ["door_noop", "door_open"]
# red_door_pos_controls = ["noop"]
# blue_door_pos_controls = ["noop"]
# # joint_policy_controls = ["noop"] # TODO: Is this factor controllable? How?

# num_controls = [
#     len(self_pos_controls),
#     len(other_pos_controls),
#     len(doors_state_controls),
#     len(red_door_pos_controls),
#     len(blue_door_pos_controls),
#     # len(joint_policy_controls),
# ]
# num_control_factors = len(num_controls)

# if VERBOSE:
#     print("\n", "-" * 10 + "Control Definition" + "-" * 10)
#     print(f"Number of control factors: {num_control_factors}")
#     print(f"Number of self position controls: {num_controls[0]}")
#     print(f"Number of other position controls: {num_controls[1]}")
#     print(f"Number of doors state controls: {num_controls[2]}")
#     print(f"Number of red door position controls: {num_controls[3]}")
#     print(f"Number of blue door position controls: {num_controls[4]}")
#     # print(f"Number of joint policy controls: {num_controls[5]}")



# "A Matrix Definition"

# """
# A (likelihood matrix) is probability of seeing an observation given a state p(o|s).
# A[i] is the likelihood matrix for the i-th modality.
# A[i] shape is [num_obs[i], num_states[0], ..., num_states[N]]

# (6,)
# (9, 9, 9, 4, 9, 9, 64)
# (9, 9, 9, 4, 9, 9, 64)
# (4, 9, 9, 4, 9, 9, 64)
# (2, 9, 9, 4, 9, 9, 64)
# (2, 9, 9, 4, 9, 9, 64)
# (49, 9, 9, 4, 9, 9, 64)

# """


# A = pymdp_utils.initialize_empty_A(num_obs, num_states)


# # --- Modality 0: Self‐Position (noisy identity on s0) ---
# base0 = np.full((num_obs[0], num_states[0]), A_NOISE_LEVEL/(num_obs[0]-1))   
# np.fill_diagonal(base0, 1.0 - A_NOISE_LEVEL)
# A[0][:] = base0.reshape(num_obs[0], num_states[0], *([1] * (A[0].ndim - 2)))


# # --- Modality 1: Other‐Position (noisy identity on s1) ---
# base1 = np.full((num_obs[1], num_states[1]), A_NOISE_LEVEL / (num_obs[1] - 1))
# np.fill_diagonal(base1, 1.0 - A_NOISE_LEVEL)
# A[1][:] = base1.reshape([num_obs[1], 1, num_states[1]] + [1] * (A[1].ndim - 3))


# # --- Modality 2: Door State (noisy identity on s2) ---
# base2 = np.full((num_obs[2], num_states[2]), A_NOISE_LEVEL/(num_obs[2]-1))
# np.fill_diagonal(base2, 1.0 - A_NOISE_LEVEL)
# A[2][:] = base2.reshape([num_obs[2], 1, 1, num_states[2]] + [1]*(A[2].ndim - 4))

# # --- Modality 3: Near red door 
# A[3].fill(1 / num_obs[3])

# # --- Modality 4: Near blue door
# A[4].fill(1 / num_obs[4])

# # --- Modality 5: Joint Action (noisy identity on s5) ---
# # A[5].fill(1 / num_obs[5])


# # my_plot_likelihood(A[5][:,0,0,0,0,0,:],yticklabels=joint_action_modality, xticklabels=joint_policy_factor)  

# if VERBOSE:
#     for m in range(num_modalities):
#         print(f"A[{m}] normalized: {pymdp_utils.is_normalized(A[m])}")


# " B Matrix Definition "

# B = pymdp_utils.initialize_empty_B(num_states, num_controls)


# # Initialize B matrices with zeros
# for f in range(num_factors):
#     B[f].fill(0.0)


# # Helper function to get next position index
# def get_next_pos_idx(pos_idx, action):
#     """Get next position index based on current position and action"""
#     # 3x3 grid mapping
#     pos_idx_to_xy = {
#         0: (0, 0),
#         1: (1, 0),
#         2: (2, 0),
#         3: (0, 1),
#         4: (1, 1),
#         5: (2, 1),
#         6: (0, 2),
#         7: (1, 2),
#         8: (2, 2),
#     }
#     xy_to_pos_idx = {v: k for k, v in pos_idx_to_xy.items()}

#     x, y = pos_idx_to_xy[pos_idx]
#     if action == 0:  # up
#         y = max(0, y - 1)
#     elif action == 1:  # down
#         y = min(2, y + 1)
#     elif action == 2:  # left
#         x = max(0, x - 1)
#     elif action == 3:  # right
#         x = min(2, x + 1)
#     elif action == 4:  # noop
#         pass  # Stay in the same position

#     return xy_to_pos_idx.get((x, y), pos_idx)


# # B[0]: Self position transitions (controlled by self movement actions)
# for current_pos in range(num_states[0]):
#     for action in range(num_controls[0]):
#         if action < 4:  # Movement actions (up, down, left, right)
#             next_pos = get_next_pos_idx(current_pos, action)
#             # Add noise to movement transitions
#             B[0][next_pos, current_pos, action] = 1.0 - B_NOISE_LEVEL
#             # Distribute noise to other possible positions
#             for other_pos in range(num_states[0]):
#                 if other_pos != next_pos:
#                     B[0][other_pos, current_pos, action] += B_NOISE_LEVEL / (
#                         num_states[0] - 1
#                     )
#         else:  # No movement action
#             B[0][current_pos, current_pos, action] = 1.0 - B_NOISE_LEVEL
#             # Distribute noise to other positions
#             for other_pos in range(num_states[0]):
#                 if other_pos != current_pos:
#                     B[0][other_pos, current_pos, action] += B_NOISE_LEVEL / (
#                         num_states[0] - 1
#                     )

# # B[1]: Other position transitions (controlled by other movement actions - mostly noop)
# for current_pos in range(num_states[1]):
#     for action in range(num_controls[1]):
#         # Other agent mostly stays in place (noop action) with noise
#         B[1][current_pos, current_pos, action] = 1.0 - B_NOISE_LEVEL
#         # Distribute noise to other positions
#         for other_pos in range(num_states[1]):
#             if other_pos != current_pos:
#                 B[1][other_pos, current_pos, action] += B_NOISE_LEVEL / (
#                     num_states[1] - 1
#                 )

# # B[2]: Door state transitions (controlled by door actions)
# door_transitions = {
#     0: [0, 1, 2],  # red_closed_blue_closed -> can open red or blue
#     1: [1, 3],  # red_open_blue_closed -> can open blue
#     2: [2, 3],  # red_closed_blue_open -> can open red
#     3: [3],  # red_open_blue_open -> terminal state
# }

# for current_door_state in range(num_states[2]):
#     for action in range(num_controls[2]):
#         if action == 0:  # noop
#             # Stay in current state with noise
#             B[2][current_door_state, current_door_state, action] = 1.0 - B_NOISE_LEVEL
#             # Distribute noise to other door states
#             for other_state in range(num_states[2]):
#                 if other_state != current_door_state:
#                     B[2][other_state, current_door_state, action] += B_NOISE_LEVEL / (
#                         num_states[2] - 1
#                     )
#         elif action == 1:  # open
#             possible_next_states = door_transitions[current_door_state]
#             prob = (1.0 - B_NOISE_LEVEL) / len(possible_next_states)
#             for next_state in possible_next_states:
#                 B[2][next_state, current_door_state, action] = prob
#             # Distribute noise to other door states
#             for other_state in range(num_states[2]):
#                 if other_state not in possible_next_states:
#                     B[2][other_state, current_door_state, action] += B_NOISE_LEVEL / (
#                         num_states[2] - len(possible_next_states)
#                     )

# # 3) Red‐Door Position transitions (factor 3)
# #    B[3].shape == (S3, S3, 1)  where S3 = num_states[3] = 9
# S3 = num_states[3]
# baseB3 = np.full((S3, S3), B_NOISE_LEVEL/(S3-1))
# np.fill_diagonal(baseB3, 1.0 - B_NOISE_LEVEL)
# # reshape to (S3, S3, 1)
# B[3][:] = baseB3.reshape(S3, S3, 1)


# # 4) Blue‐Door Position transitions (factor 4)
# #    B[4].shape == (S4, S4, 1)  where S4 = num_states[4] = 9
# S4 = num_states[4]
# baseB4 = np.full((S4, S4), B_NOISE_LEVEL/(S4-1))
# np.fill_diagonal(baseB4, 1.0 - B_NOISE_LEVEL)
# B[4][:] = baseB4.reshape(S4, S4, 1)


# # 5) Joint‐Policy transitions (factor 5)
# #    B[5].shape == (S5, S5, 1)  where S5 = num_states[5] = 64
# # S5 = num_states[5]
# # baseB5 = np.full((S5, S5), B_NOISE_LEVEL/(S5-1))
# # np.fill_diagonal(baseB5, 1.0 - B_NOISE_LEVEL)
# # B[5][:] = baseB5.reshape(S5, S5, 1)



# if VERBOSE:
#     print("\n", "-" * 10 + "B Matrix Definition " + "-" * 10)
#     print(f"B matrix shape: {[B[i].shape for i in range(len(B))]}")
#     for i, shape in enumerate([B[i].shape for i in range(len(B))]):
#         print(f"B[{i}] shape: {shape}")
#     for f in range(num_factors):
#         print(f"B[{f}] normalized: {pymdp_utils.is_normalized(B[f])}")

#     # B[0]: Self position transitions
#     for action_idx, action_name in enumerate(self_pos_controls):
#         my_plot_likelihood(
#             B[0][:, :, action_idx],
#             title=f"B[0] - Self Position Transitions ({action_name})",
#             xlabel="Current Position",
#             ylabel="Next Position",
#             xticklabels=self_pos_factor,
#             yticklabels=self_pos_factor,
#         )

#     # B[1]: Other position transitions
#     for action_idx, action_name in enumerate(other_pos_controls):
#         my_plot_likelihood(
#             B[1][:, :, action_idx],
#             title=f"B[1] - Other Position Transitions ({action_name})",
#             xlabel="Current Position",
#             ylabel="Next Position",
#             xticklabels=other_pos_factor,
#             yticklabels=other_pos_factor,
#         )

#     # B[2]: Door state transitions
#     for action_idx, action_name in enumerate(doors_state_controls):
#         my_plot_likelihood(
#             B[2][:, :, action_idx],
#             title=f"B[2] - Door State Transitions ({action_name})",
#             xlabel="Current Door State",
#             ylabel="Next Door State",
#             xticklabels=doors_state_factor,
#             yticklabels=doors_state_factor,
#         )

#     # B[3]: Red door position transitions
#     my_plot_likelihood(
#         B[3][:, :, 0],
#         title="B[3] - Red Door Position Transitions",
#         xlabel="Current Position",
#         ylabel="Next Position",
#         xticklabels=red_door_pos_factor,
#         yticklabels=red_door_pos_factor,
#     )
    
#     # B[4]: Blue door position transitions
#     my_plot_likelihood(
#         B[4][:, :, 0],
#         title="B[4] - Blue Door Position Transitions",
#         xlabel="Current Position",
#         ylabel="Next Position",
#         xticklabels=blue_door_pos_factor,
#         yticklabels=blue_door_pos_factor,
#     )   
    
#     # B[5]: Joint action transitions
#     # my_plot_likelihood( 
#     #     B[5][:, :, 0],
#     #     title="B[5] - Joint Action Transitions",
#     #     xlabel="Current Joint Action",
#     #     ylabel="Next Joint Action",
#     #     xticklabels=joint_policy_factor,
#     #     yticklabels=joint_policy_factor,
#     # )




# " C Matrix Definition  "
# C = pymdp_utils.obj_array_zeros(num_obs)



# # Self position, other position, joint action modalities
# # No strong preference → uniform or zero.
# C[0][:] = 0.0
# C[1][:] = 0.0
# # C[5][:] = 0.0

# # Door state modality
# reward_map_door_state = {
#    "red_closed_blue_closed":-0.01,
#     "red_open_blue_closed": 2.5,
#     "red_closed_blue_open": -10.0,
#     "red_open_blue_open": 10.0
# }
# for i, label in enumerate(door_state_modality):
#     C[2][i] = reward_map_door_state[label]
    
# # near red door modality
# reward_map_near_red_door = {
#     "near": 1.0,
#     "far": -0.01,
# }   
# for i, label in enumerate(near_red_door_modality):
#     C[3][i] = reward_map_near_red_door[label]

# # near blue door modality
# reward_map_near_blue_door = {
#     "near": 1.0,
#     "far": -0.01,
# }
# for i, label in enumerate(near_blue_door_modality):
#     C[4][i] = reward_map_near_blue_door[label]

# temperature = 2.0
# for m in range(len(C)):
#     C[m] = pymdp_softmax(C[m] / temperature)
    
 
 
 
# if VERBOSE: 
#     print("\n", "-" * 10 + "C Matrix Definition " + "-" * 10)
#     print(f"C shape: {[C[m].shape for m in range(len(C))]}")
#     for m in range(num_modalities):
#         print(f"C[{m}] normalized: {np.isclose(np.sum(C[0]), 1.0)}")
#         pymdp_utils.plot_beliefs(C[m], title=f"C[{m}] - Reward Beliefs")




# " D Matrix Definition "
# D1 = pymdp_utils.obj_array(num_factors)
# D1[0] = np.zeros(num_states[0])
# D1[1] = np.zeros(num_states[1])
# D1[2] = np.zeros(num_states[2])
# D1[3] = np.zeros(num_states[3])
# D1[4] = np.zeros(num_states[4])
# # D1[5] = np.zeros(num_states[5])

# D1[0][0] = 1.0
# D1[1][8] = 1.0
# D1[2][0] = 1.0  
# D1[3][:] = 1.0 / len(D1[3])
# D1[4][:] = 1.0 / len(D1[4])  # Uniform distribution over door positions
# # D1[5][:] = 1.0 / len(D1[5])  # Uniform distribution over joint actions

# if VERBOSE:
#     print("\n", "-" * 10 + "D Matrix Definition " + "-" * 10)
#     for f in range(len(D1)):
#         print(f"D[{f}] sum: {np.sum(D1[f])}, normalized: {np.isclose(np.sum(D1[f]), 1.0)}")


# "pA Matrix Definition"
# pA = pymdp_utils.dirichlet_like(A, scale=1.0)




# MODEL = {
#     "A": A,
#     "B": B,
#     "C": C,
#     "D": D1,
#     "pA": pA
# }









# def convert_obs_to_active_inference_format(obs, action_dict, agent_id)->list:
#     """
#     Converts raw PettingZoo observation to indices for the Active Inference modalities.
#     Uses variables already defined in model_2.py (global scope).
#     """
#     agents = ["agent_0", "agent_1"] 
    
#     self_agent_obs = obs[agent_id]
#     other_agent_obs= obs[next(a for a in agents if a != agent_id)]
    

#     xy_to_pos_label = {
#             (1, 1): "pos_0",
#             (2, 1): "pos_1",
#             (3, 1): "pos_2",
#             (1, 2): "pos_3",
#             (2, 2): "pos_4",
#             (3, 2): "pos_5",
#             (1, 3): "pos_6",
#             (2, 3): "pos_7",
#             (3, 3): "pos_8",
#         }




#     # --- 1. Self position ---
#     agent_pos = tuple(self_agent_obs["position"])
#     self_pos_label = xy_to_pos_label.get(agent_pos, "pos_0")
#     self_pos_idx = self_pos_modality.index(self_pos_label)
    
#      # --- 2. Other agent position ---
#     other_pos = tuple(other_agent_obs["position"])
#     other_pos_label = xy_to_pos_label.get(other_pos, "pos_0")
#     other_pos_idx = other_pos_modality.index(other_pos_label)

#     # --- 2. Door state ---
#     red = obs["red_door_opened"]
#     blue = obs["blue_door_opened"]
#     if red and blue:
#         door_state_label = "red_open_blue_open"
#     elif red:
#         door_state_label = "red_open_blue_closed"
#     elif blue:
#         door_state_label = "red_closed_blue_open"
#     else:
#         door_state_label = "red_closed_blue_closed"
#     door_state_idx = door_state_modality.index(door_state_label)

#     # --- 3. Near red door ---
#     near_red_door = self_agent_obs["near_red_door"]
#     if near_red_door:
#         near_red_door_label = "near"
#     else:
#         near_red_door_label = "far"
#     near_red_door_idx = near_red_door_modality.index(near_red_door_label)

    
#     # --- 4. Near blue door ---
#     near_blue_door = self_agent_obs["near_blue_door"]
#     if near_blue_door:
#         near_blue_door_label = "near"
#     else:
#         near_blue_door_label = "far"
#     near_blue_door_idx = near_blue_door_modality.index(near_blue_door_label)
    
#     # # --- 5. Joint action ---
#     # joint_action = (action_dict[agent_id], action_dict[next(a for a in agents if a != agent_id)])
#     # joint_action_idx = joint_action_modality.index(joint_action)
    

#     return [self_pos_idx, other_pos_idx, door_state_idx, near_red_door_idx, near_blue_door_idx]

