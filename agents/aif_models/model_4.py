"Imports"
from pymdp import utils as pymdp_utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



"Utills"




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



def my_plot_likelihood_4d(
    A, obs_modality_idx=0, state_factor_idx=0, title="", 
    xlabel="", ylabel=""
):
    """
    Plots P(obs_modality | state_factor) for specified modality and state factor.

    Args:
        A: The A matrix array
        obs_modality_idx (int): Which observation modality to plot (0-4)
        state_factor_idx (int): Which state factor to plot against (0-3)
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """
    # Get the appropriate labels based on modality and state factor
    modality_labels = {
        0: self_pos_modality,
        1: other_pos_modality, 
        2: door_state_modality,
        3: other_intention_modality,
        4: game_outcome_modality
    }
    
    state_factor_labels = {
        0: self_pos_factor,
        1: other_pos_factor,
        2: doors_state_factor, 
        3: other_intention_factor
    }
    
    yticklabels = modality_labels.get(obs_modality_idx, [])
    xticklabels = state_factor_labels.get(state_factor_idx, [])
    
    print(f"Plotting obs_modality_idx={obs_modality_idx} vs state_factor_idx={state_factor_idx}")
    
    # Extract P(obs_modality | state_factor) by taking the appropriate slice
    # A[obs_modality_idx] shape: [num_obs[obs_modality_idx], num_states[0], num_states[1], num_states[2], num_states[3]]
    # A[1][obs, self_pos, other_pos, door, intent]
    
    # Map observation modalities to their corresponding state factor axis
    modality_to_state_axis = {
        0: 1,  # self_pos obs -> self_pos state (axis 1)
        1: 2,  # other_pos obs -> other_pos state (axis 2) 
        2: 3,  # door_state obs -> door_state (axis 3)
        3: 4,  # other_intent obs -> other_intent (axis 4)
        4: 3   # game_outcome obs -> depends on door_state (axis 3)
    }
    
    modality_axis = modality_to_state_axis[obs_modality_idx]
    
    if obs_modality_idx == state_factor_idx:
        # Same modality and state factor: take the diagonal slice
        # Keep obs axis and the modality's own state axis, fix others to 0
        slice_indices = [slice(None)] + [0] * 4  # [obs, 0, 0, 0, 0]
        slice_indices[modality_axis] = slice(None)  # Keep the modality's state axis
        A_slice = A[obs_modality_idx][tuple(slice_indices)]
        print(f"Same modality: slice_indices = {slice_indices}")
    else:
        # Different modality and state factor: average over the modality's own state factor
        # Keep obs axis and target state axis, fix others to 0
        slice_indices = [slice(None)] + [0] * 4  # [obs, 0, 0, 0, 0]
        slice_indices[state_factor_idx + 1] = slice(None)  # Keep target state axis
        A_slice = A[obs_modality_idx][tuple(slice_indices)]
        print(f"Different modality: slice_indices = {slice_indices}")
        print(f"A_slice before averaging shape: {A_slice.shape}")
        # Average over the modality's own state factor
        A_slice = np.mean(A_slice, axis=1)
        print(f"A_slice after averaging shape: {A_slice.shape}")
    
    # Ensure we have a 2D array
    if A_slice.ndim == 1:
        A_slice = A_slice.reshape(-1, 1)
    elif A_slice.ndim == 3:
        # If still 3D, take the first slice of the third dimension
        A_slice = A_slice[:, :, 0]

    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        A_slice, cmap="OrRd", linewidth=1.5, xticklabels=xticklabels, yticklabels=yticklabels
    )

    plt.xlabel(xlabel or f"State Factor {state_factor_idx}")
    plt.ylabel(ylabel or f"Observation Modality {obs_modality_idx}")
    plt.title(title or f"P(Modality {obs_modality_idx} | State Factor {state_factor_idx})")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()




"Global Variables"

MAP_SIZE = "3x3"
VERBOSE = False
A_NOISE_LEVEL = 0.2




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

num_states = [len(self_pos_factor), len(other_pos_factor), len(doors_state_factor), len(other_intention_factor)]
num_factors = len(num_states)

if VERBOSE:
    print("\n","-"*10 + "States Definition" + "-"*10)
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
    print("\n","-"*10 + "Observation Definition" + "-"*10)
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

num_controls = [len(self_movement_action_names), len(other_movement_action_names), len(door_action_names), len(other_intention_action_names)]
num_control_factors = len(num_controls)

if VERBOSE:
    print("\n","-"*10 + "Control Definition" + "-"*10)
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
neutral_idx  = game_outcome_modality.index("neutral")
win_idx      = game_outcome_modality.index("win")
lose_idx     = game_outcome_modality.index("lose")
terminal_idx = doors_state_factor.index("red_open_blue_open")

# Fill modality 4 (game outcome) in nested loops
for s0 in range(num_states[0]):           # self_pos
    for s1 in range(num_states[1]):       # other_pos
        for s2 in range(num_states[2]):   # doors_state
            for s3 in range(num_states[3]):  # other_intent
                #Win
                A[4][0, s0, s1, 3, s3] = 1.0 - A_NOISE_LEVEL
                A[4][0, s0, s1, 0:3, s3] = A_NOISE_LEVEL / (num_obs[4] - 1)
                #Lose
                A[4][1, s0, s1, 2, s3] = 1.0 - A_NOISE_LEVEL
                A[4][1, s0, s1, 0:2, s3] = A_NOISE_LEVEL / (num_obs[4] - 1)
                A[4][1, s0, s1, 3, s3] = A_NOISE_LEVEL / (num_obs[4] - 1)
                #Neutral
                A[4][2, s0, s1, 0, s3] = 1.0 - A_NOISE_LEVEL
                A[4][2, s0, s1, 1, s3] = 1.0 - A_NOISE_LEVEL
                A[4][2, s0, s1, 2:4, s3] = A_NOISE_LEVEL / (num_obs[4] - 1)


for m in range(num_modalities):
    print(f"A[{m}] normalized: {pymdp_utils.is_normalized(A[m])}")
    
    
    
# my_plot_likelihood(A[4][:, :, 0,0,0], xticklabels=doors_state_factor, yticklabels=game_outcome_modality, title="Game Outcome Obs vs Game Outcome State")

# NOTE: When i get back to this, i need to test the A matrix to be correct. Then, i need to add the B matrix.






exit()
