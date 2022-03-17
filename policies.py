from mcts import MCTS
import env_util
from datetime import timedelta, datetime


def uniform_policy(env, cem, player_id, game_conf):
    action_probs = {}
    action_spaces = cem.action_spaces if cem is not None else env_util.build_action_spaces(env, game_conf.agents)
    for action in action_spaces[player_id-1]:
        action_probs[action] = 1.0 / len(action_spaces[player_id-1])
    return action_probs


def stupid_move_policy(env, cem, player_id, game_conf):
    return {(1,2): 1.0}


def maximise_cem_policy(env, cem, player_in_turn, game_conf):
    action = cem.cem_action(env, player_in_turn, game_conf.n_step)
    return { tuple(action): 1.0 }


def mcts_policy(env, cem, player_in_turn, game_conf):
    time_limit = game_conf.get_agent_by_id(player_in_turn).time_limit
    action_spaces = env_util.build_action_spaces(env, game_conf.agents)
    mcts = MCTS(env, player_in_turn, action_spaces)
    now = datetime.now()
    
    while datetime.now() - now < timedelta(seconds=time_limit):
        mcts.iterate()

    action_idx = mcts.root.best_child_idx()
    return {action_spaces[player_in_turn-1][action_idx]: 1.0}
