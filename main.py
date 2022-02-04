import os
from multi_agent_play import play
from griddly import GymWrapper, gd
from griddly_cem_agent import CEM
import visualiser
import policies
import configuration


def visualise_landscape(env, agents_confs):
    visualise_player = next(a['PlayerId'] for a in agents_confs if a['Policy'] == policies.maximise_cem_policy)
    player_id_to_names = {a['PlayerId']: (a['Name'] if 'Name' in a else a['PlayerId']) for a in agents_confs}
    empowerment_maps = visualiser.build_landscape(env, visualise_player, agents_confs, conf_obj['NStep'])
    for i, emp_map in enumerate(empowerment_maps):
        emp_pair_data = agents_confs[visualise_player-1]['EmpowermentPairs'][i]
        title = 'Empowerment: ' + player_id_to_names[emp_pair_data['Actor']] + ' -> ' + player_id_to_names[emp_pair_data['Perceptor']]
        print(title)
        print(visualiser.emp_map_to_str(emp_map))
        visualiser.plot_empowerment_landscape(env, emp_map, title)
    cem_map = {}
    for pos in empowerment_maps[0]:
        # In addition, print the CEM map that all different heatmaps weighted and summed
        cem_sum = 0
        for emp_pair_i, map in enumerate(empowerment_maps):
            cem_sum += map[pos] * agents_confs[visualise_player-1]['EmpowermentPairs'][emp_pair_i]['Weight']
        cem_map[pos] = cem_sum
    print('Weighted CEM map')
    print(visualiser.emp_map_to_str(cem_map))
    visualiser.plot_empowerment_landscape(env, cem_map, 'Weighted and summed CEM heatmap')


if __name__ == '__main__':
    USE_CONF = "threeway"
    configuration.set_verbose_calculation(True)
    configuration.activate_config(USE_CONF)
    conf_obj = configuration.active_config

    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', conf_obj.get('GriddlyDescription')),
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     image_path='./art',
                     level=0)

    env.reset()

    kbm_players = [a for a in conf_obj['Agents'] if a ['Policy'] == 'KBM']
    if len(kbm_players) > 1:
        raise Exception('Only one KBM player is supported')

    kbm_player = kbm_players[0]
    action_names = env.gdy.get_action_names()

    key_mapping = {}
    for a_i, a in enumerate(kbm_player['Actions']):
        if a == 'idle':
            key_mapping[tuple([ord(kbm_player['Keys'][a_i][0])])] = [0, 0]
            continue
        for c_i, c in enumerate(kbm_player['Keys'][a_i]):
            key_mapping[tuple([ord(c)])] = [action_names.index(a), c_i + 1]

    print('''
    Use the following keys to control the agents:
    a, w, d, s: Move
    q: Idle
    h: Heal
    space: Attack
    ''')
    
    cem = CEM(env, conf_obj['Agents'])

    play(env, agents_confs=conf_obj['Agents'], cem=cem, fps=30, zoom=3, keys_to_action=key_mapping, visualiser_callback=visualise_landscape)
