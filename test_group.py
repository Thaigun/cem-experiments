import configuration
from griddly import GymWrapper, gd
import os
from griddly_cem_agent import CEM
from random import choices
import policies
from level_generator import SimpleLevelGenerator
from datetime import datetime
from experiment_data import ExperimentData
from firebase_interface import DatabaseInterface
import env_util
import action_space_builder


RUNS_PER_CONFIG = 30


class Game:
    def __init__(self, action_space, map, cem_param):
        self.action_space = action_space
        self.map = map
        self.cem_param = cem_param
        self.done = False
        self.actions = []
        self.score = []
        self.env = None
        self.cem = None
        self.agent_policies = {}
        self.build_agent_policies()
        

    def build_agent_policies(self):
        self.agent_policies = {}
        for agent_conf in configuration.agents:
            self.agent_policies[agent_conf.player_id] = agent_conf.policy


    def play(self):
        self.create_environment()
        self.create_cem_agent()
        while not self.done:
            self.take_turn()
    
    
    def create_environment(self):
        env_file_name = configuration.griddly_description
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.env = GymWrapper(os.path.join(current_path, 'griddly_descriptions', env_file_name),
                        shader_path='shaders',
                        player_observer_type=gd.ObserverType.VECTOR,
                        global_observer_type=gd.ObserverType.SPRITE_2D,
                        image_path='./art',
                        level=0)
        self.env.reset(level_string=self.map)


    def create_cem_agent(self):
        self.cem = CEM(self.env, configuration.agents)


    def take_turn(self):
        agent_in_turn = (len(self.actions) % self.env.player_count) + 1
        action = self.select_action_for_agent(agent_in_turn)
        obs, rew, done, info = self.env.step(action)
        self.actions.append(action)
        for rew_i, reward in enumerate(rew):
            self.score[rew_i] += reward


    def select_action_for_agent(self, agent_in_turn):
        current_policy = self.agent_policies[agent_in_turn]
        action_probs = current_policy(self.env, self.cem, agent_in_turn)
        action = self.select_action_from_probs(action_probs)
        return env_util.build_action(action, self.env.player_count, agent_in_turn)


    def select_action_from_probs(self, action_probs):
        action_probs_list = list(action_probs.items())
        keys = [x[0] for x in action_probs_list]
        probs = [x[1] for x in action_probs_list]
        action = choices(keys, weights=probs)[0]
        return action


def run_test_group(conf_to_use):
    configuration.activate_config_file(conf_to_use)
    #configuration.set_visualise_all(True)
    action_space = build_action_space(2)
    cem_parameters = get_cem_parameters()
    map_parameters = get_map_parameters()
    for map_param in map_parameters:
        for _ in range(RUNS_PER_CONFIG):
            map = generate_map(map_param)
            for cem_param in cem_parameters:
                game = Game(action_space, map, cem_param)
                game.play(action_space, map, cem_param)
                save_experiment_data(action_space, map_param, map, cem_param, game.actions, game.score)


def build_action_space(player_count):
    builder = action_space_builder.ActionSpaceBuilder()
    return builder.build(player_count)


def get_cem_parameters():
    pass


def get_map_parameters():
    pass


def generate_map(map_param):
    map_config = {
        'width': 8,
        'height': 8,
        'player_count': 2,
        'bounding_obj_char': 'w',
        'obj_char_to_amount': {'w': 10, 's': 15}
    }
    level_generator = SimpleLevelGenerator(map_config)
    return level_generator.generate()


def save_experiment_data(action_space, map_param, map, cem_param, actions, score):
    pass

