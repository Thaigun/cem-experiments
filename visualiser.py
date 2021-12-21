import matplotlib.pyplot as plt
from griddly_cem_agent import find_player_pos, CEMEnv
import numpy as np
import random

def plot_empowerment_landscape(env, position_emps):
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
    plt.imshow(visu, extent=ext_visu)
    hmap = plt.imshow(emp_array, cmap='plasma', interpolation='nearest', alpha=0.5, extent=ext_map)
    plt.colorbar(hmap)
    plt.show()


def build_landscape(orig_env, player_id, emp_pairs, teams, n_step, samples=1):
    def random_move_action(env, player_id):
        available_actions = env.game.get_available_actions(player_id)
        player_pos = list(available_actions)[0]
        actions_to_ids = env.game.get_available_action_ids(player_pos, list(available_actions[player_pos]))
        # Find directions that are blocked because there are other players
        blocked_move_ids = []
        for i in range(1, env.player_count+1):
            if i == player_id:
                continue
            other_plr_pos = find_player_pos(env, i)
            other_plr_dir = [other - this for other, this in zip(other_plr_pos, player_pos)]
            if other_plr_dir == [1,0]:
                blocked_move_ids.append(3)
            elif other_plr_dir == [-1, 0]:
                blocked_move_ids.append(1)
            elif other_plr_dir == [0, 1]:
                blocked_move_ids.append(4)
            elif other_plr_dir == [0, -1]:
                blocked_move_ids.append(2)

        possible_action_combos = []
        if 'move' in actions_to_ids:
            for action_id in actions_to_ids['move']:
                if action_id not in blocked_move_ids:
                    possible_action_combos.append([env.gdy.get_action_names().index('move'), action_id])
        rnd_action = random.choice(possible_action_combos)
        full_action = [[0,0] for _ in range(env.player_count)]
        full_action[player_id-1] = rnd_action
        return full_action

    env = orig_env.clone()
    # We have to start with a random step to get an observation because the observation is not cloned.
    env.step(random_move_action(env, player_id))

    calculated_emps = [{} for _ in emp_pairs]
    for _ in range(2000):
        plr_pos = find_player_pos(env, player_id)
        if tuple(plr_pos) not in calculated_emps[0]:
            cem_env = CEMEnv(env, player_id, emp_pairs, [1, 1], teams, n_step, seed=1, samples=1)
            for emp_pair_i, emp_pair in enumerate(emp_pairs):
                state_emp = cem_env.calculate_state_empowerment(env.get_state()['Hash'], emp_pair[0], emp_pair[1])
                calculated_emps[emp_pair_i][tuple(plr_pos)] = state_emp
        # Find a random movement action that does not target a populated tile.
        # This assumes there is a 'move' action with action_ids 1=LEFT, 2=UP, 3=RIGHT, 4=DOWN.
        random_action = random_move_action(env, player_id)
        #obs[player_id-1]
        env.step(random_action)
        env.render(observer='global')
    return calculated_emps