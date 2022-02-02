import os
from multi_agent_play import play
from griddly import GymWrapper, gd
from griddly_cem_agent import CEM
import visualiser
import yaml
import policies


def maximise_cem(env, cem, player_in_turn):
    action = cem.cem_action(env, player_in_turn, conf_obj['NStep'])
    return { tuple(action): 1.0 }


def visualise_landscape(env):
    pass
    # visualise_player = 2
    # empowerment_maps = visualiser.build_landscape(env, visualise_player, conf_cem_players[visualise_player]['empowerment_pairs'], teams, n_step, conf_agent_actions, max_health)
    # for i, emp_map in enumerate(empowerment_maps):
    #     visualiser.emp_map_to_str(emp_map)
    #     visualiser.plot_empowerment_landscape(env, emp_map, 'Empowerment: ' + str(conf_cem_players[visualise_player]['empowerment_pairs'][i]))
    # cem_map = {}
    # for pos in empowerment_maps[0]:
    #     # In addition, print the CEM map that all different heatmaps weighted and summed
    #     cem_sum = 0
    #     for emp_pair_i, map in enumerate(empowerment_maps):
    #         cem_sum += map[pos] * conf_cem_players[visualise_player]['empowerment_weights'][emp_pair_i]
    #     cem_map[pos] = cem_sum
    # visualiser.emp_map_to_str(cem_map)
    # visualiser.plot_empowerment_landscape(env, cem_map, 'CEM heatmap')


if __name__ == '__main__':
    USE_CONF = "threeway"
    conf_obj = None
    with open('game_conf.yaml', 'r') as f:
        conf_obj = yaml.safe_load(f)[USE_CONF]

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
    
    agents_confs = conf_obj.get('Agents')
    # Replace the policy values with functions of the same name
    for agent_conf in agents_confs:
        agent_conf['AssumedPolicy'] = getattr(policies, agent_conf['AssumedPolicy'])
        # Agents that use the CEM policy or are controlled by a human player are special cases
        if agent_conf['Policy'] == 'CEM':
            agent_conf['Policy'] = maximise_cem
        elif agent_conf['Policy'] == 'KBM':
            pass
        else:
            agent_conf['Policy'] = getattr(policies, agent_conf['Policy'])

    cem = CEM(env, agents_confs)

    agent_policies = {}
    for agent_conf in agents_confs:
        agent_policies[agent_conf['PlayerId']] = agent_conf['Policy']

    play(env, agent_policies=agent_policies, cem=cem, fps=30, zoom=3, keys_to_action=key_mapping, visualiser_callback=visualise_landscape)
