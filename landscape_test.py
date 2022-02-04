import os
from griddly import GymWrapperFactory, gd, GymWrapper
from visualiser import plot_empowerment_landscape, build_landscape
import configuration

if __name__ == '__main__':
    configuration.activate_config("pacifist")
    conf_obj = configuration.active_config

    wrapper = GymWrapperFactory()
    name = 'projectiles_env'
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/testbed2.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    player_id = 2
    emp_pairs = [(emp['Actor'], emp['Perceptor']) for emp in conf_obj['Agents'][player_id-1]['EmpowermentPairs']]

    for nstep in range(1, 2):
        print('nstep: ', nstep)
        env.reset()
        calculated_emps = build_landscape(env, player_id, conf_obj['Agents'], nstep)
        for emp_pair_i, emp_pair in enumerate(emp_pairs):
            plot_empowerment_landscape(env, calculated_emps[emp_pair_i], 'Empowerment: ' + str(emp_pair))
    