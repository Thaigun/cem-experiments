from math import floor
import configuration
from griddly import GymWrapper, gd
import os
from griddly_cem_agent import CEM
from random import choices
import policies
from multiprocessing import Process
from level_generator import SimpleLevelGenerator
from datetime import datetime
from experiment_data import ExperimentData
from database_interface import DatabaseInterface
import time
import psutil


test_processes = []


def run_game():
    USE_CONF = 'collector'
    configuration.activate_config(USE_CONF)
    #configuration.set_visualise_all(True)
    conf_obj = configuration.active_config

    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', conf_obj.get('GriddlyDescription')),
                     shader_path='shaders',
                     player_observer_type=gd.ObserverType.VECTOR,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     image_path='./art',
                     level=0)

    map_config = {
        'width': 8,
        'height': 8,
        'player_count': 2,
        'bounding_obj_char': 'w'
    }
    obj_char_to_amount = {'w': 10, 's': 15}
    level_generator = SimpleLevelGenerator(map_config, obj_char_to_amount)
    env.reset(level_string=level_generator.generate())
    print(level_generator.level)

    agent_in_turn = 1
    agent_confs = conf_obj['Agents']
    agent_policies = {}
    for agent_conf in agent_confs:
        agent_policies[agent_conf['PlayerId']] = agent_conf['Policy']

    cem_agent_conf = [agent_conf for agent_conf in conf_obj['Agents'] if agent_conf['Policy'] == policies.maximise_cem_policy]
    cem = CEM(env, conf_obj['Agents']) if cem_agent_conf else None
    env.render(mode='human', observer='global')

    experiment_data = ExperimentData(level_generator.level, configuration.load_pure_conf(USE_CONF))

    now = datetime.now()

    done = False
    steps = 0
    cumulative_reward = [0 for _ in range(env.player_count)]
    while not done:
        current_policy = agent_policies[agent_in_turn]
        action_probs = current_policy(env, cem, agent_in_turn)
        # Select one of the keys randomly, weighted by the values
        # I'm doing it like this because I'm scared the order won't be stable if I access the keys and values separately.
        action_probs_list = list(action_probs.items())
        keys = [x[0] for x in action_probs_list]
        probs = [x[1] for x in action_probs_list]
        action = choices(keys, weights=probs)[0]
        
        full_action = [[0,0] for _ in range(env.player_count)]
        full_action[agent_in_turn-1] = list(action)
        # action_desc = env_util.action_to_str(env, action)
        # player_name = env_util.agent_id_to_name(agent_confs, agent_in_turn)
        #print(player_name, 'chose action', action_desc)
        obs, rew, done, info = env.step(full_action)
        experiment_data.add_action(full_action)
        for rew_i, reward in enumerate(rew):
            cumulative_reward[rew_i] += reward
        agent_in_turn = agent_in_turn % env.player_count + 1
        env.render(mode='human', observer='global')
        steps += 1
    print('Game finished after', steps, 'steps')
    print('Rewards: ', cumulative_reward)
    print('Time taken: ', datetime.now() - now)
    experiment_data.set_score(cumulative_reward)
    database_interface = DatabaseInterface('cem-experiments')
    database_interface.save_new_entry(experiment_data.get_data_dict())


def is_cpu_available():
    child_processes = psutil.Process().children(recursive=True)
    our_cpu_usage = sum([process.cpu_percent(interval=0.1) for process in child_processes]) / 100
    total_cpu_usage = psutil.cpu_percent(interval=0.2) / 100
    other_cpu_usage = total_cpu_usage - our_cpu_usage
    our_max_cpu_usage = 0.3 * (1-other_cpu_usage)
    cpu_bound = floor(psutil.cpu_count() * our_max_cpu_usage)
    print('Our CPU usage:', our_cpu_usage, 'total usage:', total_cpu_usage, 'other usage:', other_cpu_usage, 'our max usage:', our_max_cpu_usage, 'our bound:', cpu_bound)
    return cpu_bound > len(test_processes)


def is_memory_available():
    total_memory_used = psutil.virtual_memory().percent / 100
    child_processes = psutil.Process().children(recursive=True)
    our_usage_percentage = sum([process.memory_percent() for process in child_processes]) / 100
    other_processes_usage = total_memory_used - our_usage_percentage
    our_usable = 0.3 * (1-other_processes_usage)
    return our_usage_percentage < our_usable


def resources_available():
    return is_memory_available() and is_cpu_available()


def spawn_test_run():
    new_process = Process(target=run_game)
    new_process.start()
    print('Spawned new process:', new_process.pid)
    test_processes.append(new_process)


def clean_finished_processes():
    for process in test_processes:
        if not process.is_alive():
            print('Process', process.pid, 'finished')
            process.join()
            test_processes.remove(process)


if __name__ == '__main__':
    sleep_time = 30
    while True:
        clean_finished_processes()
        # If there are resources, reduce the sleep time a bit, and vice versa.
        if resources_available():
            sleep_time *= 0.9
            spawn_test_run()
        else:
            sleep_time /= 0.9
        # It can take a while for the memory consumption to settle, so let's wait a bit.
        time.sleep(sleep_time)
