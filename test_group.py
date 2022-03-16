import global_configuration
import game_configuration
from griddly import GymWrapper, gd
import os
from griddly_cem_agent import CEM
from random import choices
import policies
from level_generator import SimpleLevelGenerator
from firebase_interface import DatabaseInterface
import database_object
import env_util
import action_space_builder
import random


RUNS_PER_CONFIG = 30


class Game:
    def __init__(self, game_conf, map):
        self.game_conf = game_conf
        self.map = map
        self.done = False
        self.actions = []
        self.create_environment()
        self.create_cem_agent()
        self.score = [0] * self.env.player_count


    def play(self):
        while not self.done:
            self.take_turn()
    
    
    def create_environment(self):
        env_file_name = self.game_conf.griddly_description
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', env_file_name),
                        shader_path='shaders',
                        player_observer_type=gd.ObserverType.VECTOR,
                        global_observer_type=gd.ObserverType.SPRITE_2D,
                        image_path='./art',
                        level=0)
        self.env.reset(level_string=self.map)


    def create_cem_agent(self):
        self.cem = CEM(self.env, self.game_conf)


    def take_turn(self):
        agent_in_turn = (len(self.actions) % self.env.player_count) + 1
        action = self.select_action_for_agent(agent_in_turn)
        obs, rew, done, info = self.env.step(action)
        self.done = done
        self.actions.append(action)
        for rew_i, reward in enumerate(rew):
            self.score[rew_i] += reward


    def select_action_for_agent(self, agent_in_turn):
        current_policy = self.get_agent_policy(agent_in_turn)
        action_probs = current_policy(self.env, self.cem, agent_in_turn, self.game_conf)
        action = self.select_action_from_probs(action_probs)
        return env_util.build_action(action, self.env.player_count, agent_in_turn)


    def get_agent_policy(self, agent_id):
        for agent_conf in self.game_conf.agents:
            if agent_conf.player_id == agent_id:
                return agent_conf.policy


    def select_action_from_probs(self, action_probs):
        action_probs_list = list(action_probs.items())
        keys = [x[0] for x in action_probs_list]
        probs = [x[1] for x in action_probs_list]
        action = choices(keys, weights=probs)[0]
        return action


def run_test_group(conf_to_use):
    global_configuration.activate_config_file(conf_to_use)
    global_configuration.set_health_performance_consistency(True)
    #configuration.set_visualise_all(True)
    player_action_space, npc_action_space = build_action_spaces()
    db = DatabaseInterface('cem-experiments')
    game_rules_ref = save_game_rules_obj(db, player_action_space, npc_action_space)
    cem_parameters = get_cem_parameters(db)
    cem_param_keys = list(cem_parameters)
    map_parameters = get_map_parameters(db)
    map_param_keys = list(map_parameters)
    random.shuffle(cem_param_keys)
    random.shuffle(map_param_keys)
    for map_param_key in map_param_keys:
        map_param = map_parameters.get(map_param_key)
        for _ in range(RUNS_PER_CONFIG):
            map = generate_map(map_param)
            for cem_param_key in cem_param_keys:
                game_conf = build_game_conf(player_action_space, npc_action_space, cem_parameters.get(cem_param_key))
                game = Game(game_conf, map)
                game.play()
                save_experiment_data(db, game_rules_ref, cem_param_key, map_param_key, map, game.actions, game.score)


def build_action_spaces():
    builder = action_space_builder.CollectorActionSpaceBuilder()
    player_action_space = builder.build_player_action_space()
    npc_action_space = builder.build_npc_action_space()
    return list(player_action_space), list(npc_action_space)


def get_cem_parameters(db_ref):
    new_cem_param_objects = [
        database_object.CEMParamObject([2,1,2],[2,1,1],[0.5, 0.1,-0.5]),
        database_object.CEMParamObject([2,1,2],[2,1,1],[0.2, 0.5, 0.3], True),
        database_object.CEMParamObject([2,1,2],[2,1,1],[0,   0,   0]),
    ]
    cem_param_refs = fetch_data_and_save_if_none(db_ref, 'cem_params', new_cem_param_objects)
    return cem_param_refs


def get_map_parameters(db_ref):
    new_map_param_objects = [
        database_object.MapParamObject(8, 8, {'w': 10, 's': 15}),
        database_object.MapParamObject(8, 8, {'w': 4, 's': 15}),
        database_object.MapParamObject(14, 14, {'w': 30, 's': 15}),
        database_object.MapParamObject(8, 8, {'w': 10, 's': 15}),
    ]
    map_param_refs = fetch_data_and_save_if_none(db_ref, 'map_params', new_map_param_objects)
    return map_param_refs


def fetch_data_and_save_if_none(db_ref, path, default_data_objects):
    path_ref = db_ref.get_child_ref(path)
    path_data = path_ref.get()
    if path_data is None:
        for data_obj in default_data_objects:
            db_ref.save_new_entry(path, data_obj.get_data_dict())
    path_data = path_ref.get()
    return path_data


def build_game_conf(player_action_space, npc_action_space, cem_param):
    game_conf = game_configuration.GameConf('collector_game.yaml', 2)
    game_conf.add_agent('MCTS agent', 1, player_action_space, policies.mcts_policy, policies.uniform_policy)
    cem_agent_conf = game_conf.add_agent('CEM agent', 2, npc_action_space, policies.maximise_cem_policy, policies.uniform_policy)
    cem_agent_conf.set_empowerment_pairs_from_dict(cem_param['EmpowermentPairs'])
    trust_dict = cem_param['Trust']
    trust_dict['PlayerId'] = 1
    cem_agent_conf.set_trust_from_dict([trust_dict])
    return game_conf


def save_game_rules_obj(db_ref, player_action_space, npc_action_space):
    rules_obj = database_object.GameRulesObject(player_action_space, npc_action_space)
    new_item_ref = db_ref.save_new_entry('game_rules', rules_obj.get_data_dict())
    return new_item_ref


def generate_map(map_param):
    map_generator_config = {
        'width': map_param['Width'],
        'height': map_param['Height'],
        'player_count': 2,
        'bounding_obj_char': 'w',
        'obj_char_to_amount': map_param['ObjectCounts']
    }
    level_generator = SimpleLevelGenerator(map_generator_config)
    return level_generator.generate()


def save_experiment_data(db, game_rules_ref, cem_params_key, map_params_key, map, actions, score):
    game_run_ref = save_new_game_run(db, game_rules_ref.key, cem_params_key, map_params_key, map, actions, score)

    game_rules_game_runs_ref = game_rules_ref.child('game_runs')
    game_rules_game_runs_ref.push(game_run_ref.key)

    cem_params_game_runs_ref = db.get_child_ref('cem_params/' + cem_params_key + '/game_runs')
    cem_params_game_runs_ref.push(game_run_ref.key)

    map_params_game_runs_ref = db.get_child_ref('map_params/' + map_params_key + '/game_runs')
    map_params_game_runs_ref.push(game_run_ref.key)


def save_new_game_run(db, game_rules_key, cem_params_key, map_params_key, map, actions, score):
    game_run_obj = database_object.GameRunObject(map, actions, score)
    game_run_obj.set_cem_param(cem_params_key)
    game_run_obj.set_map_params(map_params_key)
    game_run_obj.set_game_rules(game_rules_key)
    game_run_ref = db.save_new_entry('game_runs', game_run_obj.get_data_dict())
    return game_run_ref
