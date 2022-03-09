def action_to_str(env, action):
    action_name = env.action_names[action[0]]
    action_desc = 'Idle' if action[1] == 0 else action_name + ' ' + env.action_input_mappings[action_name]['InputMappings'][str(action[1])]['Description']
    return action_desc


def agent_id_to_name(agents_confs, agent_id):
    return agents_confs[agent_id-1]['Name'] if 'Name' in agents_confs[agent_id-1] else 'Agent ' + str(agent_id)
    

# Returns the player_id of the player who has won.
def find_winner(info):
    if 'PlayerResults' in info:
        for plr, status in info['PlayerResults'].items():
            if status == 'Win':
                return int(plr)
    return -1


def build_action_spaces(env, agent_confs):
    def build_plr_action_space(agent_conf):
        player_action_space = []
        player_i_actions = agent_conf['Actions']
        if 'idle' in player_i_actions:
            player_action_space.append((0,0))
        for action_type_index, action_name in enumerate(env.action_names):
            if action_name in player_i_actions:
                for action_id in range(1, env.num_action_ids[action_name]):
                    player_action_space.append((action_type_index, action_id))
        return player_action_space

    action_spaces = [build_plr_action_space(agent_confs[player_i]) for player_i in range(env.player_count)] 
    return action_spaces
    

def build_action(action, player_count, player_id):
    full_action = [[0,0] for _ in range(player_count)]
    full_action[player_id-1] = list(action)
    return full_action
