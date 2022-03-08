import json
import datetime

class ExperimentData:
    def __init__(self, map, game_conf):
        self._start_time = datetime.datetime.now()
        self._map = map
        self._game_conf = game_conf
        self._actions = []
        self._score = []

    def build_data_dict(self):
        data_dict = {
            'start_time': str(self._start_time),
            'map': self._map,
            'game_conf': self._game_conf,
            'actions': self._actions,
            'score': self._score,
            'end_time': str(datetime.datetime.now())
        }
        return data_dict

    def set_score(self, score):
        self._score = score

    def add_action(self, action):
        self._actions.append(action)