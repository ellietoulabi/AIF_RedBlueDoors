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

    width, height = map(int, map_size.lower().split("x"))
    pos = []
    counter = 0
    for y in range(height):
        for x in range(width):
            pos.append(f"pos_{counter}")
            counter += 1

    return pos


def get_next_pos_idx(pos_idx, action):
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
B_NOISE_LEVEL = 0.3 #step on eachother!


AGENT_0_POLICIES = [
    # Goal-directed
    np.array(
        [
            [4, 0, 0, 0, 4],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [4, 0, 0, 0, 4],
            [5, 0, 0, 0, 5],
        ]
    ),
    np.array(
        [
            [3, 0, 0, 0, 3],
            [3, 0, 0, 0, 3],
            [4, 0, 0, 0, 4],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
        ]
    ),
    # Delayed goal-directed
    np.array(
        [
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [4, 0, 0, 0, 4],
        ]
    ),
    np.array(
        [
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [3, 0, 0, 0, 3],
            [3, 0, 0, 0, 3],
            [4, 0, 0, 0, 4],
        ]
    ),
    # Switched
    np.array(
        [
            [3, 0, 0, 0, 3],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [2, 0, 0, 0, 2],
            [4, 0, 0, 0, 4],
        ]
    ),
    np.array(
        [
            [1, 0, 0, 0, 1],
            [3, 0, 0, 0, 3],
            [3, 0, 0, 0, 3],
            [0, 0, 0, 0, 0],
            [4, 0, 0, 0, 4],
        ]
    ),
    # Ineffective
    np.array(
        [
            [3, 0, 0, 0, 3],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [4, 0, 0, 0, 4],
        ]
    ),
    np.array(
        [
            [3, 0, 0, 0, 3],
            [1, 0, 0, 0, 1],
            [2, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [4, 0, 0, 0, 4],
        ]
    ),
]


"States Definition"

self_pos_factor = grid_to_pos(MAP_SIZE)
other_pos_factor = grid_to_pos(MAP_SIZE)
red_door_pos_factor = grid_to_pos(MAP_SIZE)
blue_door_pos_factor = grid_to_pos(MAP_SIZE)
doors_state_factor = [
    "red_closed_blue_closed",
    "red_open_blue_closed",
    "red_closed_blue_open",
    "red_open_blue_open",
]

num_states = [
    len(self_pos_factor),
    len(other_pos_factor),
    len(red_door_pos_factor),
    len(blue_door_pos_factor),
    len(doors_state_factor),
]
num_factors = len(num_states)


"Observation Definition"

self_pos_modality = grid_to_pos(MAP_SIZE)
other_pos_modality = grid_to_pos(MAP_SIZE)
near_red_door_modality = ["near", "far"]
near_blue_door_modality = ["near", "far"]
doors_state_modality = [
    "red_closed_blue_closed",
    "red_open_blue_closed",
    "red_closed_blue_open",
    "red_open_blue_open",
]
num_obs = [
    len(self_pos_modality),
    len(other_pos_modality),
    len(near_red_door_modality),
    len(near_blue_door_modality),
    len(doors_state_modality),
]
num_modalities = len(num_obs)


"Controls Definition"

controls = ["up", "down", "left", "right", "open", "noop"]
num_controls = [len(controls), 1, 1, 1, len(controls)]
num_control_factors = len(num_controls)


" B Matrix Definition (Transition Probabilities)"

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
baseB2 = np.full((S2, S2), B_NOISE_LEVEL / (S2 - 1))
np.fill_diagonal(baseB2, 1.0 - B_NOISE_LEVEL)
B[2][:] = baseB2.reshape(S2, S2, 1)

S3 = num_states[3]
baseB3 = np.full((S3, S3), B_NOISE_LEVEL / (S3 - 1))
np.fill_diagonal(baseB3, 1.0 - B_NOISE_LEVEL)
B[3][:] = baseB3.reshape(S3, S3, 1)


door_transitions = {
    0: [0, 1, 2],  # red_closed_blue_closed -> can open red or blue
    1: [1, 3],  # red_open_blue_closed -> can open blue
    2: [2, 3],  # red_closed_blue_open -> can open red
    3: [3],  # red_open_blue_open -> terminal state
}

for current_door_state in range(num_states[4]):
    for action in range(num_controls[4]):
        if action in [0, 1, 2, 3, 5]:  # noop
            # Stay in current state with noise
            B[4][current_door_state, current_door_state, action] = 1.0 - B_NOISE_LEVEL
            # Distribute noise to other door states
            for other_state in range(num_states[4]):
                if other_state != current_door_state:
                    B[4][other_state, current_door_state, action] += B_NOISE_LEVEL / (
                        num_states[4] - 1
                    )
        elif action == 4:  # open
            possible_next_states = door_transitions[current_door_state]
            prob = (1.0 - B_NOISE_LEVEL) / len(possible_next_states)
            for next_state in possible_next_states:
                B[4][next_state, current_door_state, action] = prob
            # Distribute noise to other door states
            for other_state in range(num_states[4]):
                if other_state not in possible_next_states:
                    B[4][other_state, current_door_state, action] += B_NOISE_LEVEL / (
                        num_states[4] - len(possible_next_states)
                    )


" A Matrix Definition (Likelihoods)"
A = pymdp_utils.initialize_empty_A(num_obs, num_states)


base0 = np.full((num_obs[0], num_states[0]), A_NOISE_LEVEL / (num_obs[0] - 1))
np.fill_diagonal(base0, 1.0 - A_NOISE_LEVEL)
A[0][:] = base0.reshape(num_obs[0], num_states[0], *([1] * (A[0].ndim - 2)))

base1 = np.full((num_obs[1], num_states[1]), A_NOISE_LEVEL / (num_obs[1] - 1))
np.fill_diagonal(base1, 1.0 - A_NOISE_LEVEL)
A[1][:] = base1.reshape([num_obs[1], 1, num_states[1]] + [1] * (A[1].ndim - 3))

A[2].fill(0.0)
A[3].fill(0.0)
for self_pos in range(num_states[0]):
    for red_door_pos in range(num_states[2]):
        if self_pos == red_door_pos:
            A[2][
                0, self_pos, :, red_door_pos, :, :
            ] = 1.0  # "near" for all other states
        else:
            A[2][1, self_pos, :, red_door_pos, :, :] = 1.0  # "far" for all other states

for self_pos in range(num_states[0]):
    for blue_door_pos in range(num_states[3]):
        if self_pos == blue_door_pos:
            A[3][0, self_pos, :, :, blue_door_pos, :] = 1.0  # "near"
        else:
            A[3][1, self_pos, :, :, blue_door_pos, :] = 1.0  # "far"

base4 = np.full((num_obs[4], num_states[4]), A_NOISE_LEVEL / (num_obs[4] - 1))
np.fill_diagonal(base4, 1.0 - A_NOISE_LEVEL)
A[4][:] = base4.reshape(num_obs[4], 1, 1, 1, 1, num_states[4])

" C Matrix Definition (Preferences)"

C = pymdp_utils.obj_array_zeros(num_obs)
C[0][:] = 0.0
C[1][:] = 0.0

reward_map_door_state = {
    "red_closed_blue_closed": -0.01,
    "red_open_blue_closed": 5.0,
    "red_closed_blue_open": -5.0,
    "red_open_blue_open": 10.0,
}

reward_map_near_door = {
    "near": 0.,
    "far": 0.,
}

for i, label in enumerate(near_red_door_modality):
    C[2][i] = reward_map_near_door[label]

for i, label in enumerate(near_blue_door_modality):
    C[3][i] = reward_map_near_door[label]

for i, label in enumerate(doors_state_modality):
    C[4][i] = reward_map_door_state[label]

# temperature = 3.0
# for m in range(len(C)):
#     C[m] = pymdp_softmax(C[m] / temperature)


D1 = pymdp_utils.obj_array(num_factors)
D1[0] = np.zeros(num_states[0])
D1[1] = np.zeros(num_states[1])
D1[2] = np.zeros(num_states[2])
D1[3] = np.zeros(num_states[3])
D1[4] = np.zeros(num_states[4])


D1[0][0] = 1.0
D1[1][8] = 1.0
D1[2][:] = 1.0 / len(D1[2])
D1[3][:] = 1.0 / len(D1[3])
D1[4][0] = 1.0


pA = pymdp_utils.dirichlet_like(A, scale=1.0)

MODEL = {"A": A, "B": B, "C": C, "D": D1, "pA": pA, "Policies": AGENT_0_POLICIES}


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


    # --- 5. Doors state ---
    red_door_state, blue_door_state = obs["red_door_opened"], obs["blue_door_opened"]
    if red_door_state and blue_door_state:
        doors_state_label = "red_open_blue_open"
    elif red_door_state and not blue_door_state:
        doors_state_label = "red_open_blue_closed"
    elif not red_door_state and blue_door_state:
        doors_state_label = "red_closed_blue_open"
    else:
        doors_state_label = "red_closed_blue_closed"
    door_state = doors_state_modality.index(doors_state_label)
    return [self_pos_idx, other_pos_idx, near_red_door_idx, near_blue_door_idx, door_state]



