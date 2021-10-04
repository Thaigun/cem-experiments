import gym
from gym.utils.play import play
from griddly import GymWrapperFactory, gd

if __name__ == '__main__':
    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml('TestBed', 'griddly_descriptions/testbed1.yaml')
    # Match with the name of the env created with GymWrapper
    env = gym.make('GDY-TestBed-v0')
    play(env, fps=10, zoom=2)
    '''    
    env.reset()
    while True:
        env.render(observer='global')
    for step in range(1000):
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        if done:
            print("Episode finished after {} timesteps".format(step + 1))
            break
    env.close()
    '''