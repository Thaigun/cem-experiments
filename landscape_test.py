from visualiser import plot_empowerment_landscape, build_landscape
import game_configuration
import policies
from create_griddly_env import create_griddly_env

if __name__ == '__main__':
    game_config = game_configuration.GameConf('collector-game.yaml', 2, True)
    plr_conf = game_config.add_agent('Player', 1, ['lateral_move', 'collect'], policies.maximise_cem_policy, policies.uniform_policy)
    plr_conf.add_empowerment_pair(1, 1, 1.0)
    npc_conf = game_config.add_agent('NPC', 2, ['lateral_push_move'], policies.uniform_policy, policies.uniform_policy)

    env = create_griddly_env('collector_game.yaml')
    player_id = 1
    emp_pairs = [(emp.actor, emp.perceptor) for emp in game_config.agents[player_id-1].empowerment_pairs]

    for nstep in range(2, 3):
        print('nstep: ', nstep)
        env.reset()
        calculated_emps = build_landscape(env, player_id, game_config, non_blocking_objects=['star'])
        for emp_conf_i, emp_conf in enumerate(game_config.agents[player_id-1].empowerment_pairs):
            plot_empowerment_landscape(env, calculated_emps[emp_conf_i], 'Empowerment: ' + str(emp_conf))
    