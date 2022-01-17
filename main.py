import gym
import os
from multi_agent_play import play
from griddly import GymWrapper, gd
import numpy as np
import random
import empowerment_maximization
from griddly_cem_agent import CEMEnv
import visualiser
import configparser
import json

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

    
def maximise_cem(env, env_done, player_in_turn, info):
    if env_done:
        env.reset()
        return
    # Use the config object to set the variables of the CEM agent
    player_conf = conf_cem_players[player_in_turn]
    cem = CEMEnv(env, player_in_turn, player_conf['empowerment_pairs'], player_conf['empowerment_weights'], teams, n_step, conf_agent_actions, max_health=max_health)
    action = cem.cem_action()
    full_action = [[0,0] for _ in range(env.player_count)]
    full_action[player_in_turn-1] = list(action)
    obs, rew, env_done, info = env.step(full_action)
    if env_done:
        env.reset()


def visualise_landscape(env):
    empowerment_maps = visualiser.build_landscape(env, 2, [(2,2), (2,1)], teams, 1, conf_agent_actions, max_health)
    for emp_map in empowerment_maps:
        visualiser.plot_empowerment_landscape(env, emp_map)


if __name__ == '__main__':
    # Read the config file
    config = configparser.ConfigParser()
    config.read('game_conf.ini')
    env_desc = config['DEFAULT']['GriddlyDescription']
    teams = json.loads(config['DEFAULT']['Teams'])
    max_health = int(config['DEFAULT']['MaxHealth'])
    conf_cem_players = {}
    n_step = int(config['DEFAULT']['NStep'])
    conf_emp_pairs = json.loads(config['DEFAULT']['EmpowermentPairs'])
    conf_emp_weights = json.loads(config['DEFAULT']['EmpowermentWeights'])
    conf_agent_actions = json.loads(config['DEFAULT']['AgentActions'])
    for i, player_id in enumerate(json.loads(config['DEFAULT']['CEMAgents'])):
        conf_cem_players[player_id] = {
            'empowerment_pairs': conf_emp_pairs[i],
            'empowerment_weights': conf_emp_weights[i]
        }

    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', env_desc),
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

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
    play(env, fps=30, zoom=3, action_callback=maximise_cem, keys_to_action=key_mapping, visualiser_callback=visualise_landscape)
