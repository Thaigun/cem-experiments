import policies


class GameConf:
    def __init__(self, griddly_description, n_step):
        self.griddly_description = griddly_description
        self.n_step = n_step
        self.agents = []


    def set_agents_from_dict(self, agent_conf_dict):
        agent_confs = []
        for agent_conf in agent_conf_dict:
            new_agent_conf = AgentConf(agent_conf['Name'], agent_conf['PlayerId'])
            new_agent_conf.policy = getattr(policies, agent_conf['Policy'])
            new_agent_conf.assumed_policy = getattr(policies, agent_conf['AssumedPolicy'])
            if 'EmpowermentPairs' in agent_conf:
                new_agent_conf.empowerment_pairs = self.empowerment_pairs_from_dict(agent_conf['EmpowermentPairs'])
            new_agent_conf.action_space = agent_conf['Actions']
            if 'MaxHealth' in agent_conf:
                new_agent_conf.max_health = agent_conf['MaxHealth']
            if 'Trust' in agent_conf:
                new_agent_conf.trust = self.trust_conf_from_dict(agent_conf['Trust'])
            agent_confs.append(new_agent_conf)
        self.agents = agent_confs


    def empowerment_pairs_from_dict(self, empowerment_pairs_dict):
        empowerment_pairs = []
        for empowerment_pair_conf in empowerment_pairs_dict:
            empowerment_pairs.append(EmpowermentPairConf(empowerment_pair_conf['Actor'],
                                                        empowerment_pair_conf['Perceptor'],
                                                        empowerment_pair_conf['Weight']))
        return empowerment_pairs


    def trust_conf_from_dict(self, trust_conf_dict):
        trust_confs = []
        for trust_conf in trust_conf_dict:
            trust_confs.append(TrustConf(trust_conf['PlayerId'],
                                        trust_conf['Anticipation'],
                                        trust_conf['Steps']))
        return trust_confs


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

    def set_empowerment_pairs_from_dict(self, emp_pair_dict):
        empowerment_pairs = []
        for empowerment_pair_conf in emp_pair_dict:
            empowerment_pairs.append(EmpowermentPairConf(empowerment_pair_conf['Actor'],
                                                        empowerment_pair_conf['Perceptor'],
                                                        empowerment_pair_conf['Weight']))
        self.empowerment_pairs = empowerment_pairs


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
