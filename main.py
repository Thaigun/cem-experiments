import gym
from multi_agent_play import play
from griddly import GymWrapperFactory, gd
import numpy as np
import random
import empowerment_maximization
from griddly_cem_agent import CEMEnv

def make_random_move(env, env_done, info):
    available_actions = env.game.get_available_actions(2)
    if len(available_actions) == 0:
        return
    player_pos = list(available_actions)[0]
    actions_to_ids = env.game.get_available_action_ids(player_pos, list(available_actions[player_pos]))
    possible_action_combos = []
    for action_name in actions_to_ids:
        for a_id in actions_to_ids[action_name]:
            possible_action_combos.append([env.gdy.get_action_names().index(action_name), a_id])

    random_action = random.choice(possible_action_combos)
    env.step([[0,0], random_action])


def maximise_empowerment(env, env_done, info):
    if (env_done):
        return
    clone_env = env.clone()
    empowerment_agent = empowerment_maximization.EMVanillaNStepAgent(2, 2, clone_env, 1)
    action = empowerment_agent.act(clone_env)
    env.step([[0,0], action])

    
def maximise_cem(env, env_done, info):
    if env_done:
        env.reset()
        return
    cem = CEMEnv(env, 2, [(1,1), (2,2), (2,1)], [-1, 0.0, 0.1], [[1],[2]], 2, samples=1)
    action = cem.cem_action()
    obs, rew, env_done, info = env.step([[0,0], list(action)])
    if env_done:
        env.reset()

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
        (ord(' '),): [action_names.index('attack'), 1],
        }
    print('move', action_names.index('move'))
    print('heal', 4 + action_names.index('heal'))
    print('attack', 4 + action_names.index('attack'))
    clone_env = env.clone()
    play(env, fps=30, zoom=3, callback=maximise_cem, keys_to_action=key_mapping)
