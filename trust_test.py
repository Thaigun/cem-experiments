import os
from griddly import GymWrapperFactory, gd, GymWrapper
from visualiser import plot_empowerment_landscape, build_landscape, emp_map_to_str
import game_configuration

if __name__ == '__main__':
    wrapper = GymWrapperFactory()
    name = 'trust'
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/testbed_trust.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    player_id = 2

    data_object = game_configuration.load_conf_dict('trust')
    game_conf = game_configuration.game_conf_from_data_dict(data_object)

    emp_pairs = [(emp.actor, emp.perceptor) for emp in game_conf.agents[player_id-1].empowerment_pairs]
    
    print('With trust correction:')
    env.reset()
    calculated_emps = build_landscape(env, player_id, game_conf, trust_correction=True)
    for emp_pair_i, emp_pair in enumerate(emp_pairs):
        print(emp_map_to_str(calculated_emps[emp_pair_i]))
        plot_empowerment_landscape(env, calculated_emps[emp_pair_i], 'Empowerment: ' + str(emp_pair))

    print('Without trust correction:')
    data_object = game_configuration.load_conf_dict('no_trust')
    game_conf = game_configuration.game_conf_from_data_dict(data_object)
    env.reset()
    calculated_emps = build_landscape(env, player_id, game_conf, trust_correction=True)
    for emp_pair_i, emp_pair in enumerate(emp_pairs):
        print(emp_map_to_str(calculated_emps[emp_pair_i]))
        plot_empowerment_landscape(env, calculated_emps[emp_pair_i], 'Empowerment: ' + str(emp_pair))
    