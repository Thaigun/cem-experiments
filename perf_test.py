from griddly import gd, GymWrapper
import cProfile
import os

from griddly_cem_agent import CEM
import configuration

if __name__ == '__main__':
    configuration.activate_config('threeway')

    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/' + configuration.active_config.get('GriddlyDescription'),
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)
    env.reset()

    # Profile a step with CEM agent
    with cProfile.Profile() as pr:
        cem = CEM(env, configuration.active_config['Agents'])
        for _ in range(1):
            action = cem.cem_action(env, 2, 2)
            full_action = [[0, 0] for _ in range(env.player_count)]
            full_action[1] = action
            obs, rew, env_done, info = env.step(full_action)

    pr.print_stats(sort='cumtime')

    print('get_state_mapping: ', cem.calc_pd_s_a.cache_info())