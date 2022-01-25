from griddly import gd, GymWrapper
import cProfile
import os
from griddly_cem_agent import CEMEnv, EmpConf

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/testbed3.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)
    env.reset()

    # Profile a step with CEM agent
    with cProfile.Profile() as pr:
        action_sets = [['idle', 'move', 'heal', 'attack'],['idle', 'move', 'heal', 'attack']]
        configs = EmpConf([(1,1), (2,2), (2,1)], [1, 1, 1])
        cem = CEMEnv(env, {2: configs}, [[1],[2]], agent_actions=action_sets, samples=1)
        for _ in range(1):
            action = cem.cem_action(env, 2, 2)
            obs, rew, env_done, info = env.step([[0,0], list(action)])

    pr.print_stats(sort='cumtime')

    print('get_state_mapping: ', cem.get_state_mapping.cache_info())