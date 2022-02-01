def uniform_policy(action_space):
    policy = {}
    for action_idx in range(len(action_space)):
        policy[action_idx] = 1.0 / len(action_space)
    return policy


def stupid_move_policy(action_space):
    return {2: 1.0}