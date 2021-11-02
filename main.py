import gym
from multi_agent_play import play
from griddly import GymWrapperFactory, gd
import numpy as np
import random

from griddly.util.state_hash import StateHasher

def make_move(env, prev_obs, obs, action, rew, env_done, info):
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
    clone_env = env.clone()
    play(env, fps=10, zoom=2, callback=make_move, keys_to_action=key_mapping)

    '''
    env.reset()
    actions = [env.action_space.sample() for _ in range(10000)]
    for action in actions:
        obs, reward, done, info = env.step(action)
        c_obs, c_reward, c_done, c_info = clone_env.step(action)

        assert np.array_equal(obs, c_obs)
        assert reward == c_reward
        assert done == c_done
        assert info == c_info

        env_state = env.get_state()
        cloned_state = clone_env.get_state()

        env_state_hasher = StateHasher(env_state)
        cloned_state_hasher = StateHasher(cloned_state)

        env_state_hash = env_state_hasher.hash()
        cloned_state_hash = cloned_state_hasher.hash()
        assert env_state_hash == cloned_state_hash

        if done and c_done:
            env.reset()
            clone_env.reset()
    '''
