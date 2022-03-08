import configuration
from mcts import MCTS
import env_util
from datetime import timedelta, datetime


def uniform_policy(env, cem, player_id):
    action_probs = {}
    action_spaces = cem.action_spaces if cem is not None else env_util.build_action_spaces(env, configuration.active_config['Agents'])
    for action in action_spaces[player_id-1]:
        action_probs[action] = 1.0 / len(action_spaces[player_id-1])
    return action_probs


def stupid_move_policy(env, cem, player_id):
    return {(1,2): 1.0}


def maximise_cem_policy(env, cem, player_in_turn):
    action = cem.cem_action(env, player_in_turn, configuration.active_config['NStep'])
    return { tuple(action): 1.0 }


def mcts_policy(env, cem, player_in_turn):
    player_conf = next(p for p in configuration.active_config['Agents'] if p['PlayerId'] == player_in_turn)
    time_limit = player_conf['MCTSTimeLimit'] if 'MCTSTimeLimit' in player_conf else 2
    action_spaces = env_util.build_action_spaces(env, configuration.active_config['Agents'])
    mcts = MCTS(env, player_in_turn, action_spaces)
    now = datetime.now()
    
    while datetime.now() - now < timedelta(seconds=time_limit):
        mcts.iterate()

    action_idx = mcts.root.best_child_idx()
    return {action_spaces[player_in_turn-1][action_idx]: 1.0}
