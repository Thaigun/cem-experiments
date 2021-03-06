import matplotlib.pyplot as plt
from griddly_cem_agent import find_player_pos_vanilla, CEM, EnvHashWrapper
import numpy as np


def emp_map_to_str(position_emps):
    # Find the max and min x and y values from the keys of a calculated_emps dict (the coordinates that were reached)
    coordinate_list = position_emps.keys()
    max_x = max(k[0] for k in coordinate_list)
    min_x = min(k[0] for k in coordinate_list)
    max_y = max(k[1] for k in coordinate_list)
    min_y = min(k[1] for k in coordinate_list)
    # Create a 2D array representing a "heat map" of the empowerment values
    emp_array = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    for k, v in position_emps.items():
        emp_array[k[1] - min_y][k[0] - min_x] = v

    # Print out emp_array
    return_str = ''
    for row in emp_array:
        for col in row:
            return_str += "%.2f" % col + " "
        return_str += "\n"
    return return_str


def plot_empowerment_landscape(env, position_emps, title):
    # Find the max and min x and y values from the keys of a calculated_emps dict (the coordinates that were reached)
    coordinate_list = position_emps.keys()
    max_x = max(k[0] for k in coordinate_list)
    min_x = min(k[0] for k in coordinate_list)
    max_y = max(k[1] for k in coordinate_list)
    min_y = min(k[1] for k in coordinate_list)
    # Create a 2D array representing a "heat map" of the empowerment values
    emp_array = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    for k, v in position_emps.items():
        emp_array[k[1] - min_y][k[0] - min_x] = v

    ext_visu = 0, 216, 0, 216
    ext_map = 24, 216-24, 24, 216-24
    visu = env.render(observer='global', mode='rgb_array')
    plt.figure(num=title)
    plt.imshow(visu, extent=ext_visu)
    hmap = plt.imshow(emp_array, cmap='plasma', interpolation='nearest', alpha=0.5, extent=ext_map)
    plt.colorbar(hmap)
    plt.show()


def build_landscape(orig_env, player_id, game_conf, trust_correction=False, emp_idx=None, samples=1, non_blocking_objects=None):
    '''
    emp_idx: Which empowerment pair to calculate. If None, calculate all.
    '''
    def non_blocked_moves(env, player_id, visited_coords):
        available_actions = env.game.get_available_actions(player_id)
        player_pos = list(available_actions)[0]
        actions_to_ids = env.game.get_available_action_ids(player_pos, list(available_actions[player_pos]))
        # Find directions that are blocked because there are other players
        blocked_move_ids = set()
        # Coordinates in which an object would block the player
        blocking_coords = [
            tuple(sum(xy) for xy in zip(player_pos, [-1,0])),
            tuple(sum(xy) for xy in zip(player_pos, [0,-1])),
            tuple(sum(xy) for xy in zip(player_pos, [1,0])),
            tuple(sum(xy) for xy in zip(player_pos, [0,1])),
        ]
        env_state = env.get_state()
        for coord in blocking_coords:
            if coord in visited_coords:
                blocked_move_ids.add(blocking_coords.index(coord) + 1)
        for o in env_state['Objects']:
            if tuple(o['Location']) in blocking_coords and (non_blocking_objects is None or o['Name'] not in non_blocking_objects):
                blocked_move_ids.add(blocking_coords.index(tuple(o['Location'])) + 1)

        if 'move' in actions_to_ids:
            move_action = env.gdy.get_action_names().index('move')
            move_action_ids = actions_to_ids['move']
        elif 'lateral_move' in actions_to_ids:
            move_action = env.gdy.get_action_names().index('lateral_move')
            move_action_ids = actions_to_ids['lateral_move']

        possible_move_actions = []
        for action_id in move_action_ids:
            if action_id not in blocked_move_ids:
                possible_move_actions.append([move_action, action_id])
        
        return possible_move_actions

    def get_unmove_action(action):
        counter_moves = {
            1: 3,
            2: 4,
            3: 1,
            4: 2,
        }
        return (action[0], counter_moves[action[1]])

    def build_full_action(env, action, player_id):
        return [([0,0] if p != player_id else list(action)) for p in range(1, env.player_count + 1)]

    def traverse(env, player_id, calculated_emps, visited_coords, emp_pairs, cem_env, orientation_fix_action):
        plr_pos = find_player_pos_vanilla(env, player_id)
        visited_coords.add(tuple(plr_pos))
        for emp_pair_i, emp_pair in enumerate(emp_pairs):
            if emp_idx is not None and emp_idx != emp_pair_i:
                continue
            state_emp = cem_env.calculate_state_empowerment(EnvHashWrapper(env.clone()), emp_pair[0], emp_pair[1], game_conf.n_step, trust_correction=trust_correction)
            calculated_emps[emp_pair_i][tuple(plr_pos)] = state_emp
        for a in non_blocked_moves(env, player_id, visited_coords):
            env.step(build_full_action(env, a, player_id))
            env.step(orientation_fix_action)
            env.render(observer='global')
            traverse(env, player_id, calculated_emps, visited_coords, emp_pairs, cem_env, orientation_fix_action)
            env.step(build_full_action(env, get_unmove_action(a), player_id))
            env.step(orientation_fix_action)
            env.render(observer='global')

    env = orig_env.clone()
    # We have to start with an idle step because the observations are not cloned and we need them.
    env.step([[0,0] for _ in range(env.player_count)])
    # Extract the original rotation of the player with given player_id
    player_rot_name = next(o['Orientation'] for o in env.get_state()['Objects'] if o['Name'] == 'avatar' and o['PlayerId'] == player_id)
    player_rot = {'LEFT': 1, 'UP': 2, 'RIGHT': 3, 'DOWN': 4, 'NONE': 2}[player_rot_name]
    
    # Build an action that is used to fix the orientation after each step
    orientation_fix = [[0,0] for _ in range(env.player_count)]
    rotate_action_idx = env.action_names.index('rotate')
    orientation_fix[player_id-1] = [rotate_action_idx, player_rot]

    # Prepare for traversal
    player_config = game_conf.agents[player_id-1]
    emp_pairs = [(emp.actor, emp.perceptor) for emp in player_config.empowerment_pairs]
    calculated_emps = [{} for _ in player_config.empowerment_pairs]
    visited_coords = set()
    cem_env = CEM(env, game_conf, samples=samples)

    traverse(env, player_id, calculated_emps, visited_coords, emp_pairs, cem_env, orientation_fix)

    return calculated_emps