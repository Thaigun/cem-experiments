import configuration
from mcts import Node
import env_util


def uniform_policy(env, cem, player_id):
    action_probs = {}
    for action in cem.action_spaces[player_id-1]:
        action_probs[action] = 1.0 / len(cem.action_spaces[player_id-1])
    return action_probs


def stupid_move_policy(env, cem, player_id):
    return {(1,2): 1.0}


def maximise_cem_policy(env, cem, player_in_turn):
    action = cem.cem_action(env, player_in_turn, configuration.active_config['NStep'])
    return { tuple(action): 1.0 }


def mcts_policy(env, cem, player_in_turn):
    tree_root = Node()
    player_conf = next(p for p in configuration.active_config['Agents'] if p['PlayerId'] == player_in_turn)
    iter_count = player_conf['MCTSIterations'] if 'MCTSIterations' in player_conf else 30000
    action_spaces = env_util.build_action_spaces(env, configuration.active_config['Agents'])
    
    for i in range(iter_count):
        if i % 1000 == 0:
            print('Iteration: ' + str(i))
        clone_env = env.clone()
        if configuration.visualise_all:
            clone_env.render(mode='human', observer='global')
        tree_root.iterate(clone_env, player_in_turn, player_in_turn, action_spaces, max_sim_steps=20000, is_root=True)

    action_idx = tree_root.best_child_idx()
    return {action_spaces[player_in_turn-1][action_idx]: 1.0}
