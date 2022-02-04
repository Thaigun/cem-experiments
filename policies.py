import configuration

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
