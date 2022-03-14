import os
from griddly import GymWrapperFactory, gd, GymWrapper
from visualiser import plot_empowerment_landscape, build_landscape
import configuration

if __name__ == '__main__':
    configuration.activate_config_file("pacifist")

    wrapper = GymWrapperFactory()
    name = 'projectiles_env'
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(current_path + '/griddly_descriptions/testbed2.yaml',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     level=0)

    player_id = 2
    emp_pairs = [(emp.actor, emp.perceptor) for emp in configuration.agents[player_id-1].empowerment_pairs]

    for nstep in range(1, 2):
        print('nstep: ', nstep)
        env.reset()
        calculated_emps = build_landscape(env, player_id, configuration.agents, nstep)
        for emp_conf_i, emp_conf in enumerate(configuration.agents[player_id-1].empowerment_pairs):
            plot_empowerment_landscape(env, calculated_emps[emp_conf_i], 'Empowerment: ' + str(emp_conf))
    