import game_configuration
from griddly import GymWrapper, gd
from griddly_cem_agent import CEM
import policies
from level_generator import SimpleLevelGenerator
from firebase_interface import DatabaseInterface
import database_object
import env_util
import action_space_builder
import random
from datetime import datetime
from create_griddly_env import create_griddly_env


RUNS_PER_CONFIG = 30
synced_cem_params = None
synced_map_params = None


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
        self.start_time = datetime.now()
        while not self.done:
            self.take_turn()
        self.end_time = datetime.now()
    
    
    def create_environment(self):
        self.env = create_griddly_env(self.game_conf.griddly_description)
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
        action = random.choices(keys, weights=probs)[0]
        return action


def play_and_save(map, player_action_space, npc_action_space, map_param_key, cem_param_obj, game_rules_key):
    db = DatabaseInterface('cem-experiments')
    cem_param_key = list(cem_param_obj)[0]
    cem_params = cem_param_obj.get(cem_param_key)
    game_conf = build_game_conf(player_action_space, npc_action_space, cem_params)
    game = Game(game_conf, map)
    game.play()
    game_run_obj = make_game_run_obj(game_rules_key, cem_param_key, map_param_key, map, game, game_conf.griddly_description)
    game_duration = game.end_time - game.start_time
    game_run_obj.set_duration_seconds(game_duration.total_seconds())
    save_experiment_data(db, game_run_obj)


def build_game_instances():
    db = DatabaseInterface('cem-experiments')
    cem_parameters = get_cem_parameters(db)
    map_parameters = get_map_parameters(db)
    map_param_keys = list(map_parameters)
    cem_param_keys = list(cem_parameters)
    
    maps_per_param_key = {}
    for map_param_key in map_param_keys:
        map_param = map_parameters.get(map_param_key)
        maps_per_param_key[map_param_key] = generate_maps(map_param)

    skips_in_row = 0
    while skips_in_row < 100:
        player_action_space, npc_action_space = build_action_spaces()
        game_rules_ref = try_save_game_rules(db, player_action_space, npc_action_space)
        if game_rules_ref is None:
            print('SKIPPING game rules', player_action_space, npc_action_space)
            skips_in_row += 1
            continue
        skips_in_row = 0
        for map_param_key in map_param_keys:
            for map in maps_per_param_key[map_param_key]:
                for cem_param_key in cem_param_keys:
                    cem_param_obj = { cem_param_key: cem_parameters.get(cem_param_key) }
                    yield map, player_action_space, npc_action_space, map_param_key, cem_param_obj, game_rules_ref.key


def try_save_game_rules(db, player_action_space, npc_action_space):
    rules_obj = database_object.GameRulesObject(player_action_space, npc_action_space)
    rules_hash = hash(rules_obj)
    if db.child_exists('reserved_rule_hashes/' + str(rules_hash)):
        return None
    db.save_data_in_path('reserved_rule_hashes/' + str(rules_hash), rules_hash)
    game_rules_ref = db.add_new_entry('game_rules', rules_obj.get_data_dict())
    return game_rules_ref


def build_action_spaces():
    builder = action_space_builder.CollectorActionSpaceBuilder()
    player_action_space = builder.build_player_action_space()
    npc_action_space = builder.build_npc_action_space()
    return list(player_action_space), list(npc_action_space)


def get_cem_parameters(db_ref):
    new_cem_param_objects = [
        database_object.CEMParamObject([2,1,2],[2,1,1],[0.5,-0.5, 0.1]),
        database_object.CEMParamObject([2,1,2],[2,1,1],[0.2, 0.5, 0.3], True),
        database_object.CEMParamObject([2,1,2],[2,1,1],[0,   0,   0]),
    ]
    cem_param_data = fetch_data_and_save_if_none(db_ref, 'cem_params', new_cem_param_objects)
    return cem_param_data


def get_map_parameters(db_ref):
    new_map_param_objects = [
        database_object.MapParamObject(8, 8, {'w': 15, 's': 15}),
        database_object.MapParamObject(8, 8, {'w': 6, 's': 15}),
        database_object.MapParamObject(14, 14, {'w': 50, 's': 15}),
        database_object.MapParamObject(14, 14, {'w': 10, 's': 15}),
    ]
    map_param_data = fetch_data_and_save_if_none(db_ref, 'map_params', new_map_param_objects)
    return map_param_data


def fetch_data_and_save_if_none(db_ref, path, default_data_objects):
    path_ref = db_ref.get_child_ref(path)
    path_data = path_ref.get()
    if path_data is None:
        for data_obj in default_data_objects:
            db_ref.add_new_entry(path, data_obj.get_data_dict())
        path_data = path_ref.get()
    return path_data


def build_game_conf(player_action_space, npc_action_space, cem_param):
    game_conf = game_configuration.GameConf('collector_game.yaml', 2, health_performance_consistency=True)
    mcts_agent = game_conf.add_agent('MCTS agent', 1, player_action_space, policies.mcts_policy, policies.uniform_policy)
    mcts_agent.time_limit = 2
    cem_agent_conf = game_conf.add_agent('CEM agent', 2, npc_action_space, policies.maximise_cem_policy, policies.uniform_policy)
    cem_agent_conf.set_empowerment_pairs_from_dict(cem_param['EmpowermentPairs'])
    trust_dict = cem_param['Trust']
    trust_dict['PlayerId'] = 1
    cem_agent_conf.set_trust_from_dict([trust_dict])
    return game_conf


def generate_maps(map_param):
    map_generator_config = {
        'width': map_param['Width'],
        'height': map_param['Height'],
        'player_count': 2,
        'bounding_obj_char': 'w',
        'obj_char_to_amount': map_param['ObjectCounts']
    }
    level_generator = SimpleLevelGenerator(map_generator_config)
    levels = []
    for _ in range(RUNS_PER_CONFIG):
        map = level_generator.generate()
        levels.append(map)
    return levels


def make_game_run_obj(game_rules_key, cem_params_key, map_params_key, map, game, griddly_description):
    game_run_obj = database_object.GameRunObject(map, game.actions, game.score, griddly_description)
    game_run_obj.set_cem_param(cem_params_key)
    game_run_obj.set_map_params(map_params_key)
    game_run_obj.set_game_rules(game_rules_key)
    return game_run_obj


def save_experiment_data(db, game_run_obj):
    game_run_ref = save_new_game_run(db, game_run_obj)

    game_rules_game_runs_ref = db.get_child_ref('game_rules/' + game_run_obj.game_rules + '/game_runs')
    game_rules_game_runs_ref.push(game_run_ref.key)

    cem_params_game_runs_ref = db.get_child_ref('cem_params/' + game_run_obj.cem_param + '/game_runs')
    cem_params_game_runs_ref.push(game_run_ref.key)

    map_params_game_runs_ref = db.get_child_ref('map_params/' + game_run_obj.map_params + '/game_runs')
    map_params_game_runs_ref.push(game_run_ref.key)


def save_new_game_run(db, game_run_obj):
    game_run_ref = db.add_new_entry('game_runs', game_run_obj.get_data_dict())
    return game_run_ref
