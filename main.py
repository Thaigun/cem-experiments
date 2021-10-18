import gym
from multi_agent_play import play
from griddly import GymWrapperFactory, gd

def make_move(prev_obs, obs, action, rew, env_done, info):
    print('Our turn to play')

if __name__ == '__main__':
    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml('TestBed', 'griddly_descriptions/testbed1.yaml')
    # Match with the name of the env created with GymWrapper
    env = gym.make('GDY-TestBed-v0')
    play(env, fps=10, zoom=2, callback=make_move)
    