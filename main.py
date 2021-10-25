import gym
from multi_agent_play import play
from griddly import GymWrapperFactory, gd

def make_move(env, prev_obs, obs, action, rew, env_done, info):
    available_actions = env.game.get_available_actions(2)
    player_pos = list(available_actions)[0]
    print(env.game.get_available_action_ids(player_pos, list(available_actions[player_pos])))
    #random_action = available_actions.sample()
    #env.step([[0,0], random_action])

if __name__ == '__main__':
    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml('TestBed', 'griddly_descriptions/testbed1.yaml')
    # Match with the name of the env created with GymWrapper
    env = gym.make('GDY-TestBed-v0', player_observer_type=gd.ObserverType.VECTOR, global_observer_type=gd.ObserverType.SPRITE_2D)
    play(env, fps=10, zoom=2, callback=make_move)
    