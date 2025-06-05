import numpy as np
import random
import json
import os

class QLearningAgent:
    def __init__(
        self,
        agent_id,
        action_space_size,
        q_table_path=None,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.985,
        min_epsilon=0.1,
        load_existing=True
    ):
        self.agent_id = agent_id
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table_path = q_table_path
        self.agents = ["agent_0", "agent_1"]

        self.q_table = {}
        if load_existing:
            self.load_q_table()

    # def get_state(self, obs):
    #     state = []
    #     for agent_id in sorted(obs.keys()):
    #         data = obs[agent_id]
    #         pos = tuple(data['position'])
    #         red = data['red_door_opened']
    #         blue = data['blue_door_opened']
    #         near = data['near_door']
    #         state.append((agent_id, pos + (red, blue, near)))
    #     return tuple(state)

    # def get_state(self, obs):
    #     if "agent_0" not in obs or "agent_1" not in obs:
    #         return None  # or return a sentinel like "terminal"
        
    #     a0 = obs["agent_0"]
    #     a1 = obs["agent_1"]
    #     pos0 = tuple(a0["position"])
    #     pos1 = tuple(a1["position"])
    #     red = a0["red_door_opened"]
    #     blue = a0["blue_door_opened"]
    #     # near0 = a0["near_door"]
    #     # near1 = a1["near_door"]
    #     return (pos0, pos1, red, blue)
    
    def get_state(self, obs):
        if "agent_0" not in obs or "agent_1" not in obs:
            return None  # or return a sentinel like "termina
        
        
        self_agent = obs[self.agent_id]
        other_agent = obs[next(a for a in self.agents if a != self.agent_id)]
        self_pos = tuple(self_agent["position"])
        other_pos = tuple(other_agent["position"])
        red = obs["red_door_opened"]
        blue = obs["blue_door_opened"]
        self_near = self_agent["near_door"]
        other_near = other_agent["near_door"]
        
       
        door_state = None
        if red and blue:
            door_state = 3
        elif red and not blue:
            door_state = 1
        elif not red and blue:
            door_state = 2
        else:
            door_state = 0
        
        return (self_pos, door_state, self_near, other_pos)
        # return (pos0, near0, pos1, near1, red, blue)


        
    def choose_action(self, state, agent_id):
        
        if state not in self.q_table:
            return random.randint(0, self.action_space_size - 1)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        return int(np.argmax(self.q_table[state]))

    def update_q_table(self, state, actions, rewards, next_state, agent_id):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space_size) 
            
      
        current_q = self.q_table[state][actions[agent_id]]
        max_next_q = np.max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (rewards[agent_id] + self.discount_factor * max_next_q)
        self.q_table[state][actions[agent_id]] = new_q

    def decay_exploration(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self):
        if self.q_table_path:
            serializable_q = {
                str(k): v.tolist() for k, v in self.q_table.items()
            }
            with open(self.q_table_path, 'w') as file:
                json.dump(serializable_q, file)

    def load_q_table(self):
        if self.q_table_path and os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as file:
                    raw = json.load(file)
                    self.q_table = {
                        eval(k): np.array(v) for k, v in raw.items()
                    }
            except json.JSONDecodeError:
                print("Error loading Q-table. File may be empty or corrupted.")
                self.q_table = {}