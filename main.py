import gym
import os
from multi_agent_play import play
from griddly import GymWrapper, gd
import numpy as np
import random
import empowerment_maximization
from griddly_cem_agent import CEMEnv, EmpConf
import visualiser
import configparser
import json
from collections import namedtuple

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
    action = cem.cem_action(env, player_in_turn, n_step)
    full_action = [[0,0] for _ in range(env.player_count)]
    full_action[player_in_turn-1] = list(action)
    print('Agent ', str(player_in_turn), ' chose action ', action)
    obs, rew, env_done, info = env.step(full_action)
    if env_done:
        env.reset()


def visualise_landscape(env):
    visualise_player = 2
    empowerment_maps = visualiser.build_landscape(env, visualise_player, conf_cem_players[visualise_player]['empowerment_pairs'], teams, n_step, conf_agent_actions, max_health)
    for i, emp_map in enumerate(empowerment_maps):
        visualiser.emp_map_to_str(emp_map)
        visualiser.plot_empowerment_landscape(env, emp_map, 'Empowerment: ' + str(conf_cem_players[visualise_player]['empowerment_pairs'][i]))
    cem_map = {}
    for pos in empowerment_maps[0]:
        # In addition, print the CEM map that all different heatmaps weighted and summed
        cem_sum = 0
        for emp_pair_i, map in enumerate(empowerment_maps):
            cem_sum += map[pos] * conf_cem_players[visualise_player]['empowerment_weights'][emp_pair_i]
        cem_map[pos] = cem_sum
    visualiser.emp_map_to_str(cem_map)
    visualiser.plot_empowerment_landscape(env, cem_map, 'CEM heatmap')


if __name__ == '__main__':
    # Read the config file
    config = configparser.ConfigParser()
    config.read('game_conf.ini')
    active_config = config['active']
    env_desc = active_config['GriddlyDescription']
    teams = json.loads(active_config['Teams'])
    max_health = int(active_config['MaxHealth'])
    conf_cem_players = {}
    n_step = int(active_config['NStep'])
    conf_emp_pairs = json.loads(active_config['EmpowermentPairs'])
    conf_emp_weights = json.loads(active_config['EmpowermentWeights'])
    conf_agent_actions = json.loads(active_config['AgentActions'])
    conf_cem_agents = json.loads(active_config['CEMAgents'])
    for i, player_id in enumerate(conf_cem_agents):
        conf_cem_players[player_id] = {
            'empowerment_pairs': conf_emp_pairs[i],
            'empowerment_weights': conf_emp_weights[i]
        }

    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', env_desc),
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    env.reset()

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
    
    empowerment_confs = {}
    for i, player_id in enumerate(conf_cem_players):
        empowerment_confs[player_id] = (EmpConf(conf_cem_players[player_id]['empowerment_pairs'], conf_cem_players[player_id]['empowerment_weights']))

    cem = CEMEnv(env, empowerment_confs, teams, conf_agent_actions, [max_health for _ in conf_agent_actions])

    play(env, fps=30, zoom=3, action_callback=maximise_cem, keys_to_action=key_mapping, visualiser_callback=visualise_landscape)
