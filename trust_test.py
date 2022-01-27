import os
import cProfile
from griddly import GymWrapperFactory, gd, GymWrapper
from griddly_cem_agent import CEM, find_player_pos
import numpy as np
import matplotlib.pyplot as plt
import random
from visualiser import plot_empowerment_landscape, build_landscape, emp_map_to_str

if __name__ == '__main__':
    wrapper = GymWrapperFactory()
    name = 'trust'
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/testbed_trust.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    player_id = 2
    emp_pairs = [(2,2)]
    
    for nstep in range(1, 2):
        print('nstep: ', nstep)
        env.reset()
        calculated_emps = build_landscape(env, player_id, emp_pairs, [[1,2]], nstep, [['move', 'idle', 'attack'], ['move', 'idle']], 1, 1)
        for emp_pair_i, emp_pair in enumerate(emp_pairs):
            print(emp_map_to_str(calculated_emps[emp_pair_i]))
            plot_empowerment_landscape(env, calculated_emps[emp_pair_i], 'Empowerment: ' + str(emp_pair))
    