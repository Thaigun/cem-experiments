import level_generator

if __name__ == "__main__":
    map_config = {
        'width': 8,
        'height': 8,
        'bounding_obj_char': 'w',
        'player_count': 2,
        'obj_char_to_amount': {
            'w': 6,
            's': 15
        }
    }
    generator = level_generator.SimpleLevelGenerator(map_config)
    for _ in range(1):
        generator.generate()
