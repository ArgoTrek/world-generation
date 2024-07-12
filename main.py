import random
import matplotlib.pyplot as plt
import numpy as np

class WFC:
    def __init__(self, width, height, tiles, input_module, window_size=2, weights=None):
        self.width = width
        self.height = height
        self.tiles = tiles
        self.input_module = input_module
        self.window_size = window_size
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.possible_tiles = [[set(tiles) for _ in range(self.width)] for _ in range(self.height)]
        self.tile_map = {
            '1': (0, 0, 1), '2': (1, 1, 0), '3': (0, 1, 0),
            '4': (0, 0, 0.5), '5': (0.55, 0.27, 0.07), '6': (0, 0.5, 0),
            '7': (0, 0.75, 1), '8': (0.5, 0.5, 0.5), '9': (0.4, 0.4, 0.8)
        }
        self.patterns = self.extract_patterns()
        self.weights = weights if weights else {tile: 1.0 for tile in tiles}

    def extract_patterns(self):
        patterns = []
        module_height = len(self.input_module)
        module_width = len(self.input_module[0])
        for y in range(module_height - self.window_size + 1):
            for x in range(module_width - self.window_size + 1):
                pattern = tuple(
                    str(self.input_module[y + dy][x + dx])
                    for dy in range(self.window_size)
                    for dx in range(self.window_size)
                )
                patterns.append(pattern)
        return patterns

    def collapse(self):
        while not self.is_collapsed():
            x, y = self.find_min_entropy()
            if x is None or y is None:
                break
            chosen_tile = self.weighted_choice(list(self.possible_tiles[y][x]))
            self.grid[y][x] = chosen_tile
            self.propagate(x, y, chosen_tile)

    def is_collapsed(self):
        for row in self.possible_tiles:
            for cell in row:
                if len(cell) > 1:
                    return False
        return True

    def find_min_entropy(self):
        min_entropy = float('inf')
        min_pos = (None, None)
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] is None:
                    entropy = len(self.possible_tiles[y][x])
                    if entropy < min_entropy:
                        min_entropy = entropy
                        min_pos = (x, y)
        return min_pos

    def propagate(self, x, y, tile):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            current_tile = self.grid[cy][cx]
            neighbors = [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]
            for nx, ny in neighbors:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx] is None:
                        valid_patterns = set()
                        for pattern in self.patterns:
                            if self.is_valid_pattern(current_tile, pattern, cx, cy, nx, ny):
                                valid_patterns.update(pattern)
                        self.possible_tiles[ny][nx].intersection_update(valid_patterns)
                        if len(self.possible_tiles[ny][nx]) == 1:
                            new_tile = list(self.possible_tiles[ny][nx])[0]
                            self.grid[ny][nx] = new_tile
                            stack.append((nx, ny))

    def is_valid_pattern(self, current_tile, pattern, cx, cy, nx, ny):
        if nx == cx - 1 and current_tile == pattern[1]:  # Left neighbor
            return True
        if nx == cx + 1 and current_tile == pattern[0]:  # Right neighbor
            return True
        if ny == cy - 1 and current_tile == pattern[3]:  # Upper neighbor
            return True
        if ny == cy + 1 and current_tile == pattern[2]:  # Lower neighbor
            return True
        return False

    def weighted_choice(self, choices):
        total = sum(self.weights[ch] for ch in choices)
        r = random.uniform(0, total)
        upto = 0
        for ch in choices:
            w = self.weights[ch]
            if upto + w >= r:
                return ch
            upto += w

    def apply_climate_zones(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] is not None:
                    self.grid[y][x] = self.adjust_tile_for_climate(int(self.grid[y][x]), y)

    def adjust_tile_for_climate(self, tile, row):
        climate_zone = self.get_climate_zone(row)
        if climate_zone == 'temperate':
            return str(tile + 3)
        elif climate_zone == 'tundra':
            return str(tile + 6)
        return str(tile)

    def get_climate_zone(self, row):
        tundra_rows = 3 if self.height >= 9 else max(1, self.height // 6)
        temperate_rows = 4 if self.height >= 15 else max(1, self.height // 4)
        tropics_start = tundra_rows + temperate_rows
        tropics_end = self.height - tundra_rows - temperate_rows

        if row < tundra_rows or row >= self.height - tundra_rows:
            return 'tundra'
        elif row < tropics_start or row >= tropics_end:
            return 'temperate'
        else:
            return 'tropics'

    def display(self):
        data = np.zeros((self.height, self.width, 3))
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] is not None:
                    data[y, x] = self.tile_map[self.grid[y][x]]
        plt.imshow(data, interpolation='nearest')
        plt.show()

    def initialize_with_input_module(self):
        module_height = len(self.input_module)
        module_width = len(self.input_module[0])
        for y in range(module_height):
            for x in range(module_width):
                self.grid[y][x] = str(self.input_module[y][x])
                self.possible_tiles[y][x] = {str(self.input_module[y][x])}
        self.collapse()
        self.apply_climate_zones()

# Define the tiles
tiles = ['1', '2', '3']

# Input module
input_module = [
    [1, 1, 1, 1, 2, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 3, 3, 1, 2, 2, 2, 1],
    [1, 3, 3, 2, 2, 3, 3, 2],
    [1, 1, 2, 2, 2, 1, 1, 1],
    [1, 2, 2, 3, 2, 1, 1, 1],
    [1, 3, 3, 3, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

# Define the weights for each tile, making water more likely
weights = {'1': 1, '2': 0.3, '3': 0.2}

# Create a WFC instance with a parameterized sampling window size (e.g., 2 for 2x2 window)
wfc = WFC(32, 32, tiles, input_module, window_size=2, weights=weights)

# Initialize with the input module
wfc.initialize_with_input_module()

# Display the generated map
wfc.display()
