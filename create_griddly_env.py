from griddly import GymWrapper, gd
import os


def create_griddly_env(env_file_name):
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', env_file_name),
                    shader_path='shaders',
                    player_observer_type=gd.ObserverType.VECTOR,
                    global_observer_type=gd.ObserverType.SPRITE_2D,
                    image_path='./art',
                    level=0)
    return env
    