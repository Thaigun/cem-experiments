from griddly import gd, GymWrapper
import cProfile
import os

from griddly_cem_agent import CEM, find_alive_players, find_player_pos
import global_configuration

if __name__ == '__main__':
    global_configuration.activate_config_file('collector')

    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/' + global_configuration.griddly_description,
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)
    env.reset()

    # Profile a step with CEM agent
    with cProfile.Profile() as pr:
        cem = CEM(env, global_configuration.agents)
        for _ in range(1):
            action = cem.cem_action(env, 2, global_configuration.n_step)
            full_action = [[0, 0] for _ in range(env.player_count)]
            full_action[1] = action
            obs, rew, env_done, info = env.step(full_action)

    pr.print_stats(sort='tottime')

    print('get_state_mapping: ', cem.calc_pd_s_a.cache_info())
    print('find_alive_players: ', find_alive_players.cache_info())
    print('find_player_pos: ', find_player_pos.cache_info())