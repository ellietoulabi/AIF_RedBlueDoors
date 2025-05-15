from pettingzoo.utils.env import ParallelEnv
import numpy as np
import json, copy
from gym import spaces

import logging

import sys


# sys.path.append("../..")
class RedBlueDoorEnv(ParallelEnv):
    """
    Multi-Agent Red-Blue Door Environment (Ordinal Task) using PettingZoo


    RedBlueDoors is an ordinal, two-agent, non-referential game. The environment consists of a red
    door and a blue door, both initially closed. The task of agents is to open both doors, but the
    order of actions matters. The red door must be opened first, followed by the blue door.

    - A reward of **1** is given to both agents if and only if the **red door is opened first** and then the **blue door**.
    - If the **blue door is opened first**, the **episode ends immediately** with a reward of **0** for both agents.
    - The task can be solved through **visual observation** or by a **single agent opening both doors**, meaning **explicit communication is not necessary**.

    """

    metadata = {"render.modes": ["human"]}

    ACTION_MEANING = {0: "up", 1: "down", 2: "left", 3: "right", 4: "open"}

    def __init__(self, max_steps=50, config_path="configs/config.json"):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        self.cumulative_rewards = {}

        with open(config_path, "r") as file:
            config = json.load(file)
        self.config = config

        if "map" in config:
            self._parse_map(config["map"])
        else:
            self._parse_specs(config["specs"])

    # def seed(self, seed=None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     return seed

    def reset(self):
        self.red_door_opened = False
        self.blue_door_opened = False
        self.step_count = 0
        self.cumulative_rewards = {}

        if "map" in self.config:
            self._parse_map(self.config["map"])
        else:
            self._parse_specs(self.config["specs"])

        return self._get_obs(), {}

    def state(self):
        return {
            "agent_0": {
                "position": np.array(self.agent_positions["agent_0"], dtype=np.int32),
                "near_door": int(self._is_near_door(*self.agent_positions["agent_0"])),
            },
            "agent_1": {
                "position": np.array(self.agent_positions["agent_1"], dtype=np.int32),
                "near_door": int(self._is_near_door(*self.agent_positions["agent_1"])),
            },
            "red_door_opened": int(self.red_door_opened),
            "blue_door_opened": int(self.blue_door_opened),
        }

    def step(self, actions):

        rewards = {agent: -0.01 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {
            "step": None,
            "agent_0": {
                "action": None,
                "action_meaning": None,
                "reward": None,
                "cumulative_reward": None,
            },
            "agent_1": {
                "action": None,
                "action_meaning": None,
                "reward": None,
                "cumulative_reward": None,
            },
            "map": None,
        }

        for agent, action in actions.items():
            if action in [0, 1, 2, 3]:
                x, y = self.agent_positions[agent]
                dx, dy = 0, 0
                if action == 0:
                    dy = -1
                elif action == 1:
                    dy = 1
                elif action == 2:
                    dx = -1
                elif action == 3:
                    dx = 1
                new_pos = (x + dx, y + dy)
                if self._valid_move(new_pos, self.agent_positions.values()):
                    self.agent_positions[agent] = new_pos
                if self._is_near_door(
                    self.agent_positions[agent][0], self.agent_positions[agent][1]
                ):
                    rewards[agent] = 0.2
                    # print(f"Agent {agent} is near a door")

            elif action == 4:
                x, y = self.agent_positions[agent]
                if self._is_adjacent(x, y, self.red_door) and not self.red_door_opened:
                    self.red_door_opened = True
                    rewards[agent] = 0.5
                    # print("Red door opened first :)")
                elif self._is_adjacent(x, y, self.blue_door):
                    if not self.red_door_opened:
                        self.blue_door_opened = True
                        rewards = {a: -1.0 for a in self.agents}
                        terminations = {a: True for a in self.agents}
                        # self.agents = []
                        # print("Blue door opened first :(")
                    elif self.red_door_opened and (not self.blue_door_opened):
                        self.blue_door_opened = True
                        rewards = {a: 1.0 for a in self.agents}
                        terminations = {a: True for a in self.agents}
                        # self.agents = []
                        # print("Won :D")

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncations = {a: True for a in self.agents}
            rewards = {a: -1.0 for a in self.agents}

        self._update_cumulative_rewards(rewards)

        # self.agents = []
        self._fill_info(infos, actions, rewards)
        return self._get_obs(), rewards, terminations, truncations, infos

    def render(self, mode="human"):
        grid = np.full((self.height, self.width), "_", dtype=str)

        for wx, wy in self.walls:
            grid[wy, wx] = "#"

        grid[self.red_door[1], self.red_door[0]] = (
            "R" if not self.red_door_opened else "O"
        )
        grid[self.blue_door[1], self.blue_door[0]] = (
            "B" if not self.blue_door_opened else "O"
        )

        for agent, (x, y) in self.agent_positions.items():
            grid[y, x] = agent[-1]

        s = "\n".join([" ".join(row) for row in grid])
        # print(s)
        # print()
        # s = "".join("".join(row) + "\n" for row in grid)  # Ensure proper grid formatting

        return grid

    def close(self):
        pass

    def _update_cumulative_rewards(self, rewards):
        for agent, r in rewards.items():
            self.cumulative_rewards[agent] = self.cumulative_rewards.get(agent, 0.0) + r

    def _fill_info(self, infos, actions, rewards):
        infos["step"] = self.step_count
        infos["map"] = self.render()
        infos["agent_0"] = {
            "action": actions.get("agent_0", None),
            "action_meaning": self.ACTION_MEANING.get(
                actions.get("agent_0", None), "unknown"
            ),
            "reward": rewards.get("agent_0", 0.0),
            "cumulative_reward": self.cumulative_rewards.get("agent_0", 0.0),
        }
        infos["agent_1"] = {
            "action": actions.get("agent_1", None),
            "action_meaning": self.ACTION_MEANING.get(
                actions.get("agent_1", None), "unknown"
            ),
            "reward": rewards.get("agent_1", 0.0),
            "cumulative_reward": self.cumulative_rewards.get("agent_1", 0.0),
        }

    def _parse_map(self, map_data):
        """Parses a textual map representation and extracts positions."""
        self.width = len(map_data[0].split())
        self.height = len(map_data)
        self.walls = set()
        self.agent_positions = {}
        self.red_door = None
        self.blue_door = None

        for y, row in enumerate(map_data):
            row = row.split()
            for x, char in enumerate(row):
                if char == "#":
                    self.walls.add((x, y))
                elif char == "R":
                    self.red_door = (x, y)
                elif char == "B":
                    self.blue_door = (x, y)
                elif char == "0":
                    self.agent_positions["agent_0"] = (x, y)
                elif char == "1":
                    self.agent_positions["agent_1"] = (x, y)

        self._initialize_common_attributes()

    def _parse_specs(self, config):
        """Parses specifications-based environment setup."""
        self.width = config["width"]
        self.height = config["height"]
        self.red_door = tuple(config["red_door"])
        self.blue_door = tuple(config["blue_door"])
        self.walls = set(tuple(wall) for wall in config["walls"])
        self.agent_positions = {
            "agent_0": tuple(config["agent_positions"][0]),
            "agent_1": tuple(config["agent_positions"][1]),
        }

        self._initialize_common_attributes()

    def _initialize_common_attributes(self):
        """Initialize shared attributes across both map and specs parsing."""
        self.agents = list(self.agent_positions.keys())

        self.red_door_opened = False
        self.blue_door_opened = False

        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "position": spaces.Box(
                        low=0,
                        high=max(self.width, self.height),
                        shape=(2,),
                        dtype=np.int32,
                    ),
                    "red_door_opened": spaces.Discrete(2),
                    "blue_door_opened": spaces.Discrete(2),
                }
            )
            for agent in self.agents
        }

    def _valid_move(self, pos, occupied):
        return (
            pos not in self.walls
            and pos not in occupied
            and pos != self.red_door
            and pos != self.blue_door
        )

    def _is_near_door(self, x, y):
        # neighbors = [(x + dx, y + dy)
        #              for dx in [-1, 0, 1]
        #              for dy in [-1, 0, 1]
        #              if not (dx == 0 and dy == 0)]
        # return any(pos in neighbors for pos in [self.red_door, self.blue_door])
        neighbors = [
            (x, y - 1),  # up
            (x, y + 1),  # down
            (x - 1, y),  # left
            (x + 1, y),  # right
        ]
        return any(pos in neighbors for pos in [self.red_door, self.blue_door])

    def _is_adjacent(self, x, y, target):
        tx, ty = target
        return (abs(x - tx) == 1 and y == ty) or (abs(y - ty) == 1 and x == tx)

    def _get_obs(self):
        return {
            "agent_0": {
                "position": np.array(self.agent_positions["agent_0"], dtype=np.int32),
                "near_door": int(self._is_near_door(*self.agent_positions["agent_0"])),
            },
            "agent_1": {
                "position": np.array(self.agent_positions["agent_1"], dtype=np.int32),
                "near_door": int(self._is_near_door(*self.agent_positions["agent_1"])),
            },
            "red_door_opened": int(self.red_door_opened),
            "blue_door_opened": int(self.blue_door_opened),
        }
