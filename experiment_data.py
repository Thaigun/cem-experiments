import json
import datetime

class ExperimentData:
    def __init__(self, map, game_conf):
        self.data_dict = {
            'start_time': str(datetime.datetime.now()),
            'map': map,
            'game_conf': game_conf,
            'actions': [],
            'score': []
        }

    def get_data_dict(self):
        self.data_dict['end_time'] = str(datetime.datetime.now())
        return self.data_dict

    def set_score(self, score):
        self.data_dict['score'] = score

    def add_action(self, action):
        self.data_dict['actions'].append(action)
