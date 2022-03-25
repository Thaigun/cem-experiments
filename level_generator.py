from random import choices
from griddly.util.rllib.environment.level_generator import LevelGenerator
import copy

class SimpleLevelGenerator(LevelGenerator):
    def __init__(self, config):
        super().__init__(config)
        # A dictionary mapping object characters to the amount of that object in the level
        self.object_amounts = copy.deepcopy(config['obj_char_to_amount'])

        self.width = config.get('width', 10)
        self.height = config.get('height', 10)
        self.bounding_obj_char = config.get('bounding_obj_char', 'w')
        
        player_count = config.get('player_count', 2)
        for player_id in range(1, player_count+1):
            self.object_amounts['P' + str(player_id)] = 1
        
        self.object_amounts['.'] = self.height * self.width - sum(self.object_amounts.values())
        self.level = ""

    def generate(self):
        valid_candidate = False
        while not valid_candidate:
            level_sting_lines = [[] for _ in range(self.height + 2)]
            objects_left = copy.copy(self.object_amounts)
            for y in range(0, self.width + 2):
                for x in range(0, self.height + 2):
                    if x == 0 or x == self.width + 1 or y == 0 or y == self.height + 1:
                        level_sting_lines[y].append(self.bounding_obj_char)
                    else:
                        tile_obj = self._pseudo_random_object(objects_left)
                        level_sting_lines[y].append(tile_obj)
                        objects_left[tile_obj] -= 1
            valid_candidate = self.test_lateral_connectedness(level_sting_lines) and self.test_diagonal_connectedness(level_sting_lines)
        self.level = "\n".join([" ".join(line) for line in level_sting_lines])
        return self.level

    def _pseudo_random_object(self, objects_left):
        # Returns a random object character, but with a probability proportional to the amount of that object
        # in the level.
        random_object = choices(list(objects_left.keys()), weights=list(objects_left.values()))[0]
        return random_object

    def test_lateral_connectedness(self, level_string_lines):
        # Assumes that the bounding character is the only blocking character
        current_pos = None
        for y, line in enumerate(level_string_lines):
            for x, char in enumerate(line):
                if char != self.bounding_obj_char:
                    current_pos = (x, y)
                    break
            if current_pos is not None:
                break
        # Depth-first search to find all squares that are not equal to the bounding_obj_char
        visited = set([current_pos])
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self._find_connected_traversables(current_pos, level_string_lines, visited, directions)
        return len(visited) == self.width * self.height - self.object_amounts[self.bounding_obj_char]

    def test_diagonal_connectedness(self, level_string_lines):
        def find_starting_square(level_str_lines, find_str):
            for y, line in enumerate(level_str_lines):
                for x, char in enumerate(line):
                    if char == find_str:
                        return (x, y)

        starting_squares = [find_starting_square(level_string_lines, 'P1'), find_starting_square(level_string_lines, 'P2')]
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for starting_square in starting_squares:
            visited = set([starting_square])
            self._find_connected_traversables(starting_square, level_string_lines, visited, directions)
            free_square_count = self.width * self.height - self.object_amounts[self.bounding_obj_char]
            if len(visited) < free_square_count * 0.35:
                print('Rejected level because of too many unconnected squares')
                print(len(visited), '<', free_square_count)
                return False
        return True
            

    def _find_connected_traversables(self, current_pos, level_string_lines, visited, directions):
        for direction in directions:
            new_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            if new_pos in visited:
                continue
            if level_string_lines[new_pos[1]][new_pos[0]] == self.bounding_obj_char:
                continue
            visited.add(new_pos)
            self._find_connected_traversables(new_pos, level_string_lines, visited, directions)
