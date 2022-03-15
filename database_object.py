class GameRunObject:
    def __init__(self, map, actions, score):
        self.map = map
        self.actions = actions
        self.score = score
        self.cem_param = ''
        self.game_rules = ''
        self.map_params = ''

    def set_cem_param(self, cem_param):
        self.cem_param = cem_param

    def set_game_rules(self, game_rules):
        self.game_rules = game_rules

    def set_map_params(self, map_params):
        self.map_params = map_params

    def get_data_dict(self):
        return {
            'Map': self.map,
            'Actions': self.actions,
            'Score': self.score,
            'CemParams': self.cem_param,
            'GameRules': self.game_rules,
            'MapParams': self.map_params
        }


class GameRulesObject:
    def __init__(self, player_action_space, npc_action_space):
        self.player_action_space = player_action_space
        self.npc_action_space = npc_action_space
        self.game_runs = []

    def get_data_dict(self):
        return {
            'PlayerActions': self.player_action_space,
            'NpcActions': self.npc_action_space,
            'GameRuns': self.game_runs
        }


class CEMParamObject:
    def __init__(self, actors, perceptors, weights):
        assert len(actors) == len(perceptors) == len(weights)
        self.empowerment_pairs = [EmpowermentPairObject(actors[i], perceptors[i], weights[i]) for i in range(len(actors))]
        self.game_runs = []

    def get_data_dict(self):
        return {
            'EmpowermentPairs': [empowerment_pair.get_data_dict() for empowerment_pair in self.empowerment_pairs],
        }


class EmpowermentPairObject:
    def __init__(self, actor, perceptor, weight):
        self.actor = actor
        self.perceptor = perceptor
        self.weight = weight

    def get_data_dict(self):
        return {
            'Actor': self.actor,
            'Perceptor': self.perceptor,
            'Weight': self.weight
        }


class MapParamObject:
    def __init__(self, width, height, object_counts):
        self.width = width
        self.height = height
        self.object_counts = object_counts
        self.game_runs = []

    def get_data_dict(self):
        return {
            'Width': self.width,
            'Height': self.height,
            'ObjectCounts': self.object_counts,
            'GameRuns': self.game_runs
        }