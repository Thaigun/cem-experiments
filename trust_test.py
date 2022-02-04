import os
from griddly import GymWrapperFactory, gd, GymWrapper
from visualiser import plot_empowerment_landscape, build_landscape, emp_map_to_str
import configuration

if __name__ == '__main__':
    wrapper = GymWrapperFactory()
    name = 'trust'
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/testbed_trust.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    player_id = 2
    configuration.activate_config("trust")
    emp_pairs = [(emp['Actor'], emp['Perceptor']) for emp in configuration.active_config['Agents'][player_id-1]['EmpowermentPairs']]
    
    for nstep in range(1, 2):
        print('nstep: ', nstep)
        env.reset()
        trust_correction_steps = (False, True)
        calculated_emps = build_landscape(env, player_id, configuration.active_config['Agents'], configuration.active_config['NStep'], trust_correction=True)
        for emp_pair_i, emp_pair in enumerate(emp_pairs):
            print(emp_map_to_str(calculated_emps[emp_pair_i]))
            plot_empowerment_landscape(env, calculated_emps[emp_pair_i], 'Empowerment: ' + str(emp_pair))

    