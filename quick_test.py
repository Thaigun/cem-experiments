import os
import cProfile
from griddly import GymWrapperFactory, gd, GymWrapper
from griddly_cem_agent import CEMEnv
import numpy as np

if __name__ == '__main__':
    wrapper = GymWrapperFactory()

    name = 'projectiles_env'

    current_path = os.path.dirname(os.path.realpath(__file__))

    env = GymWrapper(current_path + '/griddly_descriptions/testbed1.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    env.reset()

    pr = cProfile.Profile()
    pr.enable()

    cem_env = CEMEnv(env, 2, [(1,1), (2,2), (2,1)], [1, 1, 1], 2, 1)
    selected_action = cem_env.cem_action()
    print(selected_action)

    pr.disable()
    pr.create_stats()
    pr.print_stats(sort='cumtime')

    env = GymWrapper(current_path + '/griddly_descriptions/testbed2.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    env.reset()