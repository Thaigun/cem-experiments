from random import choices
from griddly.util.rllib.environment.level_generator import LevelGenerator
import copy

class SimpleLevelGenerator(LevelGenerator):
    def __init__(self, config):
        super().__init__(config)
        # A dictionary mapping object characters to the amount of that object in the level
        self.object_amounts = config['obj_char_to_amount']

        self.width = config.get('width', 10)
        self.height = config.get('height', 10)
        self.bounding_obj_char = config.get('bounding_obj_char', 'w')
        
        player_count = config.get('player_count', 2)
        for player_id in range(1, player_count+1):
            self.object_amounts['P' + str(player_id)] = 1
        
        self.object_amounts['.'] = self.height * self.width - sum(self.object_amounts.values())
        self.level = ""

    def generate(self):
        # Width and height do not contain the bounding walls
        total_size = (self.width + 2) * (self.height + 2)
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
        self.level = "\n".join([" ".join(line) for line in level_sting_lines])
        return self.level

    def _pseudo_random_object(self, objects_left):
        # Returns a random object character, but with a probability proportional to the amount of that object
        # in the level.
        random_object = choices(list(objects_left.keys()), weights=list(objects_left.values()))[0]
        return random_object