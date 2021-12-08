import os
import cProfile
from griddly import GymWrapperFactory, gd, GymWrapper
from griddly_cem_agent import CEMEnv
import numpy as np

if __name__ == '__main__':
    wrapper = GymWrapperFactory()

    name = 'projectiles_env'

    current_path = os.path.dirname(os.path.realpath(__file__))

    env = GymWrapper(current_path + '/griddly_descriptions/testbed2.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    env.reset()

    #pr = cProfile.Profile()
    #pr.enable()
    player_id = 1
    calculated_emps = {}

    for _ in range(1000):
        plr_2_pos = next(o['Location'] for o in env.get_state()['Objects'] if o['Name'] == 'plr' and o['PlayerId'] == player_id)
        if tuple(plr_2_pos) not in calculated_emps:
            cem_env = CEMEnv(env, player_id, [(player_id,player_id)], [1], [[1], [2]], 2, 1)
            state_emp = cem_env.calculate_state_empowerment(env.get_state()['Hash'], player_id, player_id)
            calculated_emps[tuple(plr_2_pos)] = state_emp
        #Sample an action, but only for one player
        action_sample = env.action_space.sample()
        action = [0] * env.player_count
        action[player_id-1] = action_sample[player_id-1]
        
        obs, rew, done, info = env.step(action)
        
    # Find the max and min x and y values from the keys of the  calculated_emps dict
    max_x = max(k[0] for k in calculated_emps.keys())
    min_x = min(k[0] for k in calculated_emps.keys())
    max_y = max(k[1] for k in calculated_emps.keys())
    min_y = min(k[1] for k in calculated_emps.keys())
    # Create a 2D array of the calculated_emps dict
    emp_array = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    for k, v in calculated_emps.items():
        emp_array[k[1] - min_y][k[0] - min_x] = v

    # Print out emp_array
    for row in emp_array:
        for col in row:
            print("%.2f" % col, end=' ')
        print()

    #pr.disable()
    #pr.create_stats()
    #pr.print_stats(sort='cumtime')

    env = GymWrapper(current_path + '/griddly_descriptions/testbed2.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    env.reset()