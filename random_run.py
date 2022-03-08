import gym
import griddly
from griddly import GymWrapper, gd
import os
import time
from level_generator import SimpleLevelGenerator

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/collector_game.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     image_path='./art',
                     level=0)

    map_config = {
        'width': 8,
        'height': 8,
        'player_count': 2,
        'bounding_obj_char': 'w'
    }
    obj_char_to_amount = {'w': 10, 's': 15}
    level_generator = SimpleLevelGenerator(map_config, obj_char_to_amount)
    env.reset(level_string=level_generator.generate())
    
    for s in range(10000):
        obs, reward, done, info = env.step(env.action_space.sample())

        env.render(observer='global')
        time.sleep(0.1)
        if done:
            env.reset(level_string=level_generator.generate())