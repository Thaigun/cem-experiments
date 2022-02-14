import os
from multi_agent_play import play
from griddly import GymWrapper, gd
from griddly_cem_agent import CEM
import visualiser
import policies
import configuration
import env_util


def visualise_landscape(env, agents_confs, emp_idx=None, trust_correction=False):
    '''
    emp_idx: 
    None -> visualise all
    -1 -> visualise the weighted full cem
    >0 -> visualise the cem pair with the given index
    '''
    # Find the player id of the agent with CEM policy
    visualise_player = next(a['PlayerId'] for a in agents_confs if a['Policy'] == policies.maximise_cem_policy)
    # Make a mapping from player id to the name defined in YAML
    player_id_to_names = {a['PlayerId']: (a['Name'] if 'Name' in a else a['PlayerId']) for a in agents_confs}
    empowerment_maps = visualiser.build_landscape(env, visualise_player, agents_confs, configuration.active_config['NStep'], trust_correction, None if emp_idx == -1 else emp_idx)
    # TODO: If empowerment_maps is a dict of dicts, does the enumerate work anymore?
    for i, emp_map in enumerate(empowerment_maps):
        if i != emp_idx and emp_idx is not None:
            continue
        emp_pair_data = agents_confs[visualise_player-1]['EmpowermentPairs'][i]
        title = 'Empowerment: ' + player_id_to_names[emp_pair_data['Actor']] + ' -> ' + player_id_to_names[emp_pair_data['Perceptor']] + ', steps: ' + str(configuration.active_config['NStep'])
        print(title)
        print(visualiser.emp_map_to_str(emp_map))
        visualiser.plot_empowerment_landscape(env, emp_map, title)
    
    if emp_idx is None or emp_idx == -1:
        cem_map = {}
        for pos in empowerment_maps[0]:
            # In addition, print the CEM map that all different heatmaps weighted and summed
            cem_sum = 0
            for emp_pair_i, map in enumerate(empowerment_maps):
                cem_sum += map[pos] * agents_confs[visualise_player-1]['EmpowermentPairs'][emp_pair_i]['Weight']
            cem_map[pos] = cem_sum
        print('Weighted CEM map; steps: ' + str(configuration.active_config['NStep']))
        print(visualiser.emp_map_to_str(cem_map))
        visualiser.plot_empowerment_landscape(env, cem_map, 'Weighted and summed CEM heatmap, steps: ' + str(configuration.active_config['NStep']))


if __name__ == '__main__':
    USE_CONF = "collector"
    configuration.set_verbose_calculation(True)
    configuration.activate_config(USE_CONF)
    conf_obj = configuration.active_config

    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', conf_obj.get('GriddlyDescription')),
                     shader_path='shaders',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     image_path='./art',
                     level=0)

    env.reset()

    cem_agent_conf = [agent_conf for agent_conf in conf_obj['Agents'] if agent_conf['Policy'] == policies.maximise_cem_policy]
    if cem_agent_conf:
        print("Use the following keys to print different empowerments")
        for emp_conf_i, emp_conf in enumerate(cem_agent_conf['EmpowermentPairs']):
            print(f'{emp_conf_i + 1}: {env_util.agent_id_to_name(conf_obj["Agents"], emp_conf["Actor"])} -> {env_util.agent_id_to_name(conf_obj["Agents"], emp_conf["Perceptor"])}')
        print("0: Weighted full-CEM")
        print("P: All")
    
    print("Press T to toggle trust correction on or off for the visualisation (DOESN'T APPLY TO AGENT DECISIONS)")
    print("Press Y to toggle health-performance consistency on or off (APPLIES TO AGENT DECISIONS)")

    action_names = env.gdy.get_action_names()
    reserved_keys = ['p', 't', 'y', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    kbm_players = [a for a in conf_obj['Agents'] if a ['Policy'] == 'KBM']
    if len(kbm_players) > 1:
        raise Exception('Only one KBM player is supported')

    key_mapping = None
    if len(kbm_players) == 1:
        kbm_player = kbm_players[0]
        key_mapping = {}
        key_action_pairs = zip(kbm_player['Keys'], kbm_player['Actions'])
        print('Use the following keys to control the agents:')
        for key, action_name in key_action_pairs:
            print(f'{key}: {action_name}')
            if action_name == 'idle':
                if key in reserved_keys:
                    raise Exception('Ill conf. Reserved key: ' + key)
                idle_key = ord(key)
                key_mapping[tuple([idle_key])] = [0, 0]
                continue
            for c_i, c in enumerate(key):
                if c in reserved_keys:
                    raise Exception('Ill conf. Reserved key: ' + c)
                key_mapping[tuple([ord(c)])] = [action_names.index(action_name), c_i + 1]
    
    cem = CEM(env, conf_obj['Agents']) if cem_agent_conf else None

    play(env, agents_confs=conf_obj['Agents'], cem=cem, fps=30, zoom=3, keys_to_action=key_mapping, visualiser_callback=visualise_landscape)
