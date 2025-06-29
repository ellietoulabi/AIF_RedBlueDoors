"Utills"
def grid_to_pos(map_size: int) -> list:
    width, height = map(int, map_size.lower().split("x"))
    pos = []
    counter = 0
    for y in range(height):
        for x in range(width):
            pos.append(f"pos_{counter}")
            counter += 1

    return pos  


"Global Variables"
MAP_SIZE = "3x3"






"States Definition"

self_pos_factor = grid_to_pos(MAP_SIZE)
print(self_pos_factor)