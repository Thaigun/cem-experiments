from firebase_interface import DatabaseInterface
from dotenv import dotenv_values
import os
from griddly import GymWrapper, gd
import time
import env_util
import test_group
from griddly_cem_agent import CEM
import global_configuration


def get_db():
    config = dotenv_values('.env')
    root_path = config['PLAYBACK_ROOT']
    return DatabaseInterface('cem-experiments', root_path=root_path)


def get_game_run_data(run_id):
    db = get_db()
    game_run_ref = db.get_child_ref('game_runs/' + run_id)
    game_run_data = game_run_ref.get()
    return game_run_data


def create_rerun_env(griddly_file_name):
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', griddly_file_name),
                    shader_path='shaders',
                    player_observer_type=gd.ObserverType.VECTOR,
                    global_observer_type=gd.ObserverType.SPRITE_2D,
                    image_path='./art',
                    level=0)
    return env


def print_env_information(game_run, game_conf):
    print_game_rules(game_conf)
    print_cem_params(game_conf)
    print_score(game_run)


def print_game_rules(game_conf):
    print('Game Rules:')
    print('Player actions:', game_conf.agents[0].action_space)
    print('NPC actions:', game_conf.agents[1].action_space)


def print_cem_params(game_conf):
    print('CEM Params:')
    print('Empowerment pairs: ')
    for emp_pair in game_conf.agents[1].empowerment_pairs:
        print(str(emp_pair))
    print('Trust: ', game_conf.agents[1].trust)


def print_score(game_run_data):
    print('Score: ', game_run_data['Score'])


def build_game_conf(game_run):
    db = get_db()
    game_rules_key = game_run['GameRules']
    cem_params_key = game_run['CemParams']
    game_rules_data = db.get_child_ref('game_rules/' + game_rules_key).get()
    cem_params_data = db.get_child_ref('cem_params/' + cem_params_key).get()
    player_actions = game_rules_data['PlayerActions']
    npc_actions = game_rules_data['NpcActions']
    return test_group.build_game_conf(player_actions, npc_actions, cem_params_data)


def replay_game(env, actions, delay, cem=None):
    env.render(observer='global')
    if cem is not None:
        global_configuration.set_verbose_calculation(True)
    agent_idx = 0
    for action in actions:
        if cem is not None and agent_idx == 1:
            cem.cem_action(env, 2, 2)
        env.step(action)
        player_name = 'plr' if agent_idx == 0 else 'npc'
        print(player_name, env_util.action_to_str(env, action[agent_idx]))
        env.render(observer='global')
        time.sleep(delay)
        agent_idx = (agent_idx + 1) % 2


def run_replay_for_id(run_id, delay=2, calculate_emps=False):
    game_run_data = get_game_run_data(run_id)
    env = create_rerun_env(game_run_data['GriddlyDescription'])
    env.reset(level_string=game_run_data['Map'])
    game_conf = build_game_conf(game_run_data)
    print_env_information(game_run_data, game_conf)
    cem = CEM(env, game_conf) if calculate_emps else None
    replay_game(env, game_run_data['Actions'], delay=delay, cem=cem)


if __name__=='__main__':
    run_id = input('Please enter the run id: ')
    emp_calc = input('Calculate empowerment pairs? (y/n)').lower() == 'y'
    run_replay_for_id(run_id, calculate_emps=emp_calc)
    input('Press enter to exit')
