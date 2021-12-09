import os
import cProfile
from griddly import GymWrapperFactory, gd, GymWrapper
from griddly_cem_agent import CEMEnv, find_player_pos
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    wrapper = GymWrapperFactory()

    name = 'projectiles_env'

    current_path = os.path.dirname(os.path.realpath(__file__))

    env = GymWrapper(current_path + '/griddly_descriptions/testbed2.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    player_id = 1
    emp_pairs = [(1,1), (1,2)]
    
    for nstep in range(1, 3):
        print('nstep: ', nstep)
        env.reset()
        #pr = cProfile.Profile()
        #pr.enable()
        calculated_emps = [{} for _ in emp_pairs]

        for _ in range(1000):
            env.render(observer='global')
            plr_pos = find_player_pos(env.get_state(), player_id)
            if tuple(plr_pos) not in calculated_emps[0]:
                cem_env = CEMEnv(env, player_id, emp_pairs, [1, 1], [[1], [2]], nstep, seed=1, samples=1)
                for emp_pair_i, emp_pair in enumerate(emp_pairs):
                    state_emp = cem_env.calculate_state_empowerment(env.get_state()['Hash'], emp_pair[0], emp_pair[1])
                    calculated_emps[emp_pair_i][tuple(plr_pos)] = state_emp
            #Sample an action, but only for one player
            action_sample = env.action_space.sample()
            action = [0] * env.player_count
            action[player_id-1] = action_sample[player_id-1]
            obs, rew, done, info = env.step(action)
            
        for emp_pair_i, emp_pair in enumerate(emp_pairs):
            # Find the max and min x and y values from the keys of a calculated_emps dict (the coordinates that were reached)
            coordinate_list = calculated_emps[emp_pair_i].keys()
            max_x = max(k[0] for k in coordinate_list)
            min_x = min(k[0] for k in coordinate_list)
            max_y = max(k[1] for k in coordinate_list)
            min_y = min(k[1] for k in coordinate_list)
            # Create a 2D array representing a "heat map" of the empowerment values
            emp_array = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
            for k, v in calculated_emps[emp_pair_i].items():
                emp_array[k[1] - min_y][k[0] - min_x] = v

            # Print out emp_array
            print(emp_pair)
            for row in emp_array:
                for col in row:
                    print("%.2f" % col, end=' ')
                print()
            
            ext_visu = 0, 216, 0, 216
            ext_map = 24, 216-24, 24, 216-24
            visu = env.render(observer='global', mode='rgb_array')
            plt.imshow(visu, extent=ext_visu)
            hmap = plt.imshow(emp_array, cmap='plasma', interpolation='nearest', alpha=0.5, extent=ext_map)
            plt.colorbar(hmap)
            plt.show()

    #input('Press enter to exit')
    #pr.disable()
    #pr.create_stats()
    #pr.print_stats(sort='cumtime')
    