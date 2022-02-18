import configuration
from griddly import GymWrapper, gd
import os
from griddly_cem_agent import CEM
from random import choices
import env_util
import policies
from multiprocessing import Pool, Process


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

    env.reset()

    agent_in_turn = 1
    agent_confs = conf_obj['Agents']
    agent_policies = {}
    for agent_conf in agent_confs:
        agent_policies[agent_conf['PlayerId']] = agent_conf['Policy']

    cem_agent_conf = [agent_conf for agent_conf in conf_obj['Agents'] if agent_conf['Policy'] == policies.maximise_cem_policy]
    cem = CEM(env, conf_obj['Agents']) if cem_agent_conf else None
    env.render(mode='human', observer='global')

    done = False
    steps = 0
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
        action_desc = env_util.action_to_str(env, action)
        player_name = env_util.agent_id_to_name(agent_confs, agent_in_turn)
        #print(player_name, 'chose action', action_desc)
        obs, rew, done, info = env.step(full_action)
        agent_in_turn = agent_in_turn % env.player_count + 1
        env.render(mode='human', observer='global')
        steps += 1
    print('Game finished after', steps, 'steps')


if __name__ == '__main__':
    processes = []
    for _ in range(1):
        p = Process(target=run_game)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    