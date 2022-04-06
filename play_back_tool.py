from firebase_interface import DatabaseInterface
from dotenv import dotenv_values
import os
from griddly import GymWrapper, gd
import time
import env_util


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


def print_env_information(game_run_data):
    db = get_db()
    print_game_rules(game_run_data, db)
    print_cem_params(game_run_data, db)
    print_score(game_run_data)


def print_game_rules(game_run_data, db):
    game_rules_key = game_run_data['GameRules']
    game_rules_data = db.get_child_ref('game_rules/' + game_rules_key).get()
    print('Game Rules:')
    print('Player actions:', game_rules_data['PlayerActions'])
    print('NPC actions:', game_rules_data['NpcActions'])


def print_cem_params(game_run_data, db):
    cem_params_key = game_run_data['CemParams']
    cem_params_data = db.get_child_ref('cem_params/' + cem_params_key).get()
    print('CEM Params:')
    print('Empowerment pairs: ', cem_params_data['EmpowermentPairs'])
    print('Trust: ', cem_params_data['Trust'])


def print_score(game_run_data):
    print('Score: ', game_run_data['Score'])


def replay_game(env, actions, delay):
    env.render(observer='global')
    agent_idx = 0
    for action in actions:
        env.step(action)
        player_name = 'plr' if agent_idx == 0 else 'npc'
        print(player_name, env_util.action_to_str(env, action[agent_idx]))
        env.render(observer='global')
        time.sleep(delay)
        agent_idx = (agent_idx + 1) % 2


def run_replay_for_id(run_id, delay=2):
    game_run_data = get_game_run_data(run_id)
    env = create_rerun_env(game_run_data['GriddlyDescription'])
    env.reset(level_string=game_run_data['Map'])
    print_env_information(game_run_data)
    replay_game(env, game_run_data['Actions'], delay=delay)


if __name__=='__main__':
    run_id = input('Please enter the run id: ')
    run_replay_for_id(run_id)
    input('Press enter to exit')
