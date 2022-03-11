import datetime


class GameRunObject:
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.map = ''
        self.actions = []
        self.score = []
        self.cem_param = ''
        self.action_sets = ''
        self.map_params = ''


class GameRulesObject:
    def __init__(self):
        self.action_spaces = []
        self.game_runs = []


class CEMParamObject:
    def __init__(self):
        self.actor = 0
        self.perceptor = 0
        self.weight = 0.0
        self.game_runs = []


class MapParamObject:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.object_counts = {}
        self.game_runs = []
