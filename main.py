import os
from multi_agent_play import play
from griddly import GymWrapper, gd
from griddly_cem_agent import CEM
import visualiser
import policies
import conf_parser


def visualise_landscape(env, agents_confs):
    visualise_player = next(a['PlayerId'] for a in agents_confs if a['Policy'] == policies.maximise_cem_policy)
    empowerment_maps = visualiser.build_landscape(env, visualise_player, agents_confs, conf_obj['NStep'])
    for i, emp_map in enumerate(empowerment_maps):
        visualiser.emp_map_to_str(emp_map)
        visualiser.plot_empowerment_landscape(env, emp_map, 'Empowerment: ' + str(agents_confs[visualise_player-1]['EmpowermentPairs'][i]))
    cem_map = {}
    for pos in empowerment_maps[0]:
        # In addition, print the CEM map that all different heatmaps weighted and summed
        cem_sum = 0
        for emp_pair_i, map in enumerate(empowerment_maps):
            cem_sum += map[pos] * agents_confs[visualise_player-1]['EmpowermentPairs'][emp_pair_i]['Weight']
        cem_map[pos] = cem_sum
    visualiser.emp_map_to_str(cem_map)
    visualiser.plot_empowerment_landscape(env, cem_map, 'CEM heatmap')


if __name__ == '__main__':
    USE_CONF = "threeway"
    conf_parser.activate_config(USE_CONF)
    conf_obj = conf_parser.active_config

    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', conf_obj.get('GriddlyDescription')),
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
        # Idle
        (ord('q'),): [0, 0],
        # Rest of the actions don't have a direction for now
        (ord('h'),): [action_names.index('heal'), 1],
        (ord(' '),): [action_names.index('attack'), 1],
        }

    print('''
    Use the following keys to control the agents:
    a, w, d, s: Move
    q: Idle
    h: Heal
    space: Attack
    ''')
    
    cem = CEM(env, conf_obj['Agents'])

    play(env, agents_confs=conf_obj['Agents'], cem=cem, fps=30, zoom=3, keys_to_action=key_mapping, visualiser_callback=visualise_landscape)
