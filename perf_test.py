from griddly import GymWrapperFactory, gd, GymWrapper
import cProfile
import os
from griddly_cem_agent import CEMEnv

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/testbed1.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)
    env.reset()

    # Profile a step with CEM agent
    with cProfile.Profile() as pr:
        cem = CEMEnv(env, 2, [(1,1), (2,2), (2,1)], [1, 1, 1], [[1],[2]], 2, samples=1)
        action = cem.cem_action()
        obs, rew, env_done, info = env.step([[0,0], list(action)])

    pr.print_stats(sort='cumtime')