def action_to_str(env, action):
    action_name = env.action_names[action[0]]
    action_desc = 'Idle' if action[1] == 0 else action_name + ' ' + env.action_input_mappings[action_name]['InputMappings'][str(action[1])]['Description']
    return action_desc