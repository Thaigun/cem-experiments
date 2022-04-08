import policies
import yaml


class GameConf:
    def __init__(self, griddly_description, n_step, health_performance_consistency=False):
        self.griddly_description = griddly_description
        self.n_step = n_step
        self.agents = []
        self.health_performance_consistency = health_performance_consistency
        self.keys = None


    def get_agent_by_id(self, agent_id):
        for agent in self.agents:
            if agent.player_id == agent_id:
                return agent
        return None


    def set_agents_from_dict(self, agent_conf_dict):
        agent_confs = []
        for agent_conf in agent_conf_dict:
            agent_name = agent_conf['Name'] if 'Name' in agent_conf else 'Agent' + str(agent_conf['PlayerId'])
            new_agent_conf = AgentConf(agent_name, agent_conf['PlayerId'])
            new_agent_conf.action_space = agent_conf['Actions']
            if agent_conf['Policy'] != 'KBM':
                new_agent_conf.policy = getattr(policies, agent_conf['Policy'])
            else:
                new_agent_conf.policy = 'KBM'
            new_agent_conf.assumed_policy = getattr(policies, agent_conf['AssumedPolicy'])
            if 'EmpowermentPairs' in agent_conf:
                new_agent_conf.set_empowerment_pairs_from_dict(agent_conf['EmpowermentPairs'])
            new_agent_conf.action_space = agent_conf['Actions']
            if 'MaxHealth' in agent_conf:
                new_agent_conf.max_health = agent_conf['MaxHealth']
            if 'Trust' in agent_conf:
                new_agent_conf.set_trust_from_dict(agent_conf['Trust'])
            if 'TimeLimit' in agent_conf:
                new_agent_conf.time_limit = agent_conf['TimeLimit']
            if 'Keys' in agent_conf:
                new_agent_conf.keys = agent_conf['Keys']
            agent_confs.append(new_agent_conf)
        self.agents = agent_confs


    def add_agent(self, name, agent_id, action_space, policy, assumed_policy):
        self.agents.append(AgentConf(name, agent_id))
        self.agents[-1].action_space = action_space
        self.agents[-1].policy = policy
        self.agents[-1].assumed_policy = assumed_policy
        return self.agents[-1]


class AgentConf:
    def __init__(self, name, player_id):
        self.name = name
        self.player_id = player_id
        self.policy = None
        self.assumed_policy = None
        self.empowerment_pairs = None
        self.action_space = []
        self.max_health = 2
        self.trust = None
        self.time_limit = 2

    def set_empowerment_pairs_from_dict(self, emp_pair_dict):
        empowerment_pairs = []
        for empowerment_pair_conf in emp_pair_dict:
            empowerment_pairs.append(EmpowermentPairConf(empowerment_pair_conf['Actor'],
                                                        empowerment_pair_conf['Perceptor'],
                                                        empowerment_pair_conf['Weight']))
        self.empowerment_pairs = empowerment_pairs

    def set_trust_from_dict(self, trust_dict):
        trust_confs = []
        for trust_conf in trust_dict:
            trust_confs.append(TrustConf(trust_conf['PlayerId'],
                                        trust_conf['Anticipation'],
                                        trust_conf['Steps'] if 'Steps' in trust_conf else []))
        self.trust = trust_confs

    def add_empowerment_pair(self, actor, perceptor, weight):
        if self.empowerment_pairs is None:
            self.empowerment_pairs = []
        self.empowerment_pairs.append(EmpowermentPairConf(actor, perceptor, weight))


class EmpowermentPairConf:
    def __init__(self, actor, perceptor, weight):
        self.actor = actor
        self.perceptor = perceptor
        self.weight = weight

    def __str__(self):
        return f'({self.actor} -> {self.perceptor}) (weight: {self.weight})'


class TrustConf:
    def __init__(self, player_id, anticipation, steps):
        self.player_id = player_id
        self.anticipation = anticipation
        self.steps = steps


def load_conf_dict(conf_name):
    with open('game_conf.yaml', 'r') as f:
        return yaml.safe_load(f)[conf_name]


def game_conf_from_data_dict(data_object):
    health_performance_consistency = data_object['HealthPerformanceConsistency'] if 'HealthPerformanceConsistency' in data_object else False
    game_conf = GameConf(data_object['GriddlyDescription'],
                         data_object['NStep'],
                         health_performance_consistency)
    game_conf.set_agents_from_dict(data_object['Agents'])
    return game_conf