import gym
from multi_agent_play import play
from griddly import GymWrapperFactory, gd
import pprint

def make_move(env, prev_obs, obs, action, rew, env_done, info):
    available_actions = env.game.get_available_actions(2)
    if len(available_actions) == 0:
        return
    player_pos = list(available_actions)[0]
    print(env.game.get_available_action_ids(player_pos, list(available_actions[player_pos])))
    game_state = env.get_state()
    print([item for item in game_state['Objects'] if item['Name'] == 'plr'])
    #random_action = available_actions.sample()
    #env.step([[0,0], random_action])

if __name__ == '__main__':
    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml('TestBed', 'griddly_descriptions/testbed1.yaml')
    # Match with the name of the env created with GymWrapper
    env = gym.make('GDY-TestBed-v0', player_observer_type=gd.ObserverType.VECTOR, global_observer_type=gd.ObserverType.SPRITE_2D)
    action_names = env.gdy.get_action_names()
    key_mapping = {
        # Move actions are action_type 0, the first four are the action_ids for move (directions)
        (ord('a'),): [action_names.index('move'), 1], 
        (ord('w'),): [action_names.index('move'), 2],
        (ord('d'),): [action_names.index('move'), 3],
        (ord('s'),): [action_names.index('move'), 4],
        # No-op may be implemented later?
        (ord('q'),): [0, 0],
        # Rest of the actions don't have a direction for now
        (ord('h'),): [action_names.index('heal'), 1],
        (ord(' '),): [action_names.index('melee'), 1],
        (ord('e'),): [action_names.index('ranged'), 1],
        }
    
    play(env, fps=10, zoom=2, callback=make_move, keys_to_action=key_mapping)
    