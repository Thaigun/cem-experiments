import os
import cProfile
from griddly import GymWrapperFactory, gd, GymWrapper
from griddly_cem_agent import CEMEnv, find_player_pos
import numpy as np
import matplotlib.pyplot as plt
import random
from visualiser import plot_empowerment_landscape, build_landscape

if __name__ == '__main__':
    wrapper = GymWrapperFactory()

    name = 'projectiles_env'

    current_path = os.path.dirname(os.path.realpath(__file__))

    env = GymWrapper(current_path + '/griddly_descriptions/testbed2.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    player_id = 2
    emp_pairs = [(2,2), (2,1)]
    
    for nstep in range(1, 2):
        print('nstep: ', nstep)
        env.reset()
        calculated_emps = build_landscape(env, player_id, emp_pairs, [[1], [2]], nstep, [['move', 'idle'], ['move', 'idle']], False, 1)
        for emp_pair_i, emp_pair in enumerate(emp_pairs):
            plot_empowerment_landscape(env, calculated_emps[emp_pair_i], 'Empowerment: ' + str(emp_pair))
    