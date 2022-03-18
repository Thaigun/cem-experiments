import multi_agent_play
from griddly_cem_agent import CEM
import visualiser
import policies
import global_configuration
import env_util
from create_griddly_env import create_griddly_env
import game_configuration


def visualise_landscape(env, game_conf, emp_idx=None, trust_correction=False):
    '''
    emp_idx: 
    None -> visualise all
    -1 -> visualise the weighted full cem
    >0 -> visualise the cem pair with the given index
    '''
    agents_confs = game_conf.agents
    # Find the player id of the agent with CEM policy
    visualise_player = next(a.player_id for a in agents_confs if a.policy == policies.maximise_cem_policy)
    empowerment_maps = visualiser.build_landscape(env, visualise_player, game_conf, trust_correction, None if emp_idx == -1 else emp_idx)
    # TODO: If empowerment_maps is a dict of dicts, does the enumerate work anymore?
    for i, emp_map in enumerate(empowerment_maps):
        if i != emp_idx and emp_idx is not None:
            continue
        emp_pair_data = agents_confs[visualise_player-1].empowerment_pairs[i]
        title = 'Empowerment: ' + env_util.agent_id_to_name(agents_confs, emp_pair_data.actor) + ' -> ' + env_util.agent_id_to_name(agents_confs, emp_pair_data.perceptor) + ', steps: ' + str(game_conf.n_step)
        print(title)
        print(visualiser.emp_map_to_str(emp_map))
        visualiser.plot_empowerment_landscape(env, emp_map, title)
    
    if emp_idx is None or emp_idx == -1:
        cem_map = {}
        for pos in empowerment_maps[0]:
            # In addition, print the CEM map that all different heatmaps weighted and summed
            cem_sum = 0
            for emp_pair_i, map in enumerate(empowerment_maps):
                cem_sum += map[pos] * agents_confs[visualise_player-1].empowerment_pairs[emp_pair_i].weight
            cem_map[pos] = cem_sum
        print('Weighted CEM map; steps: ' + str(game_conf.n_step))
        print(visualiser.emp_map_to_str(cem_map))
        visualiser.plot_empowerment_landscape(env, cem_map, 'Weighted and summed CEM heatmap, steps: ' + str(game_conf.n_step))


if __name__ == '__main__':
    USE_CONF = "fight"
    conf_dict = game_configuration.load_conf_dict(USE_CONF)
    game_conf = game_configuration.game_conf_from_data_dict(conf_dict)
    global_configuration.set_verbose_calculation(True)

    env = create_griddly_env(game_conf.griddly_description)
    env.reset()

    cem_agent_conf = [agent_conf for agent_conf in game_conf.agents if agent_conf.policy == policies.maximise_cem_policy]
    if cem_agent_conf:
        cem_agent_conf = cem_agent_conf[0]
        print("Use the following keys to print different empowerments")
        for emp_conf_i, emp_conf in enumerate(cem_agent_conf.empowerment_pairs):
            print(f'{emp_conf_i + 1}: {env_util.agent_id_to_name(game_conf.agents, emp_conf.actor)} -> {env_util.agent_id_to_name(game_conf.agents, emp_conf.perceptor)}')
        print("0: Weighted full-CEM")
        print("P: All")
    
    print("Press T to toggle trust correction on or off for the visualisation (DOESN'T APPLY TO AGENT DECISIONS)")
    print("Press Y to toggle health-performance consistency on or off (APPLIES TO AGENT DECISIONS)")

    kbm_players = [a for a in game_conf.agents if a.policy == 'KBM']
    if len(kbm_players) > 1:
        raise Exception('Only one KBM player is supported')

    action_names = env.gdy.get_action_names()
    reserved_keys = ['p', 't', 'y', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    key_mapping = None
    if len(kbm_players) == 1:
        kbm_player = kbm_players[0]
        key_mapping = {}
        key_action_pairs = zip(kbm_player.keys, kbm_player.action_space)
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
    
    cem = CEM(env, game_conf) if cem_agent_conf else None

    multi_agent_play.play(env, game_conf, cem=cem, fps=30, zoom=3, keys_to_action=key_mapping, visualiser_callback=visualise_landscape)
