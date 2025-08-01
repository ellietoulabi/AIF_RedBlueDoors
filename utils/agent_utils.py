# def find_paths_to_door_on_map(
#     maps={
#         'map_1': [
#             "# # # # #",
#             "# 0 _ _ B",
#             "# _ _ _ #",
#             "R _ _ 1 #",
#             "# # # # #"
#         ],
#         'map_2': [
#             "# # # R #",
#             "# 0 _ _ #",
#             "B _ _ _ #",
#             "# _ _ 1 #",
#             "# # # # #"
#         ]
#     },
#     agent_id=0,
#     state_factors_and_controls={
#         'self_pos':['up', 'down', 'left', 'right', 'noop'], 
#         'other_pos':['noop'], 
#         'doors_state':['door_noop', 'open'], 
#         'red_door_pos':['noop'], 
#         'blue_door_pos':['noop'], 
#         'joint_policy':['noop', 'open']
#         },
   
#     policy_len=5,
#     num_policies=25
# ):
    
    
#     from collections import deque

#     moves = {
#         'up':    (0, -1),
#         'down':  (0, 1),
#         'left':  (-1, 0),
#         'right': (1, 0)
#     }
#     width = len(map_lines[0])
#     height = len(map_lines)

#     # Find agent and door positions
#     for y, row in enumerate(map_lines):
#         for x, c in enumerate(row):
#             if c == agent_char:
#                 start_pos = (x, y)
#             if c == door_char:
#                 door_pos = (x, y)

#     # Helper: is cell passable?
#     def is_passable(x, y):
#         if not (0 <= x < width and 0 <= y < height):
#             return False
#         c = map_lines[y][x]
#         return c in ['_', '0', '1']

#     # Helper: is adjacent to door?
#     def is_adjacent_to_door(x, y):
#         for dx, dy in moves.values():
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < width and 0 <= ny < height and (nx, ny) == door_pos:
#                 return True
#         return False

#     queue = deque()
#     queue.append((start_pos, []))
#     visited = set()
#     results = []

#     while queue:
#         pos, path = queue.popleft()
#         if (pos, tuple(path)) in visited or len(path) > policy_len:
#             continue
#         visited.add((pos, tuple(path)))

#         x, y = pos
#         # If adjacent to door, can open
#         if is_adjacent_to_door(x, y) and len(path) < policy_len:
#             results.append(path + [open_action])
#             continue

#         for action in actions:
#             dx, dy = moves[action]
#             nx, ny = x + dx, y + dy
#             if is_passable(nx, ny):
#                 queue.append(((nx, ny), path + [action]))

#     # Only keep paths of length ≤ policy_len
#     results = [seq for seq in results if len(seq) <= policy_len]
#     return results



# map_lines = [
#     "# # # # #",
#     "# 0 _ _ B",
#     "# _ _ _ #",
#     "R _ _ 1 #",
#     "# # # # #"
# ]
# # Remove spaces for easier indexing
# map_lines = [row.replace(' ', '') for row in map_lines]

# paths = find_paths_to_door_on_map(map_lines, agent_char='0', door_char='R', policy_len=6)
# for p in paths:
#     print(p)