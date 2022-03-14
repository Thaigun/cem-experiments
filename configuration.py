import yaml
import policies
import env_util

verbose_calculation = False
health_performance_consistency = True
visualise_all = False
n_step = 1
griddly_description = ""
agents = []


class AgentConf:
    def __init__(self, name, player_id):
        self.name = name
        self.player_id = player_id
        self.policy = None
        self.assumed_policy = None
        self.empowerment_pairs = None
        self.actions = []
        self.max_health = 0


class EmpowermentpairConf:
    def __init__(self, actor, perceptor, weight):
        self.actor = actor
        self.perceptor = perceptor
        self.weight = weight

    def __str__(self):
        return f'{env_util.agent_id_to_name(agents, self.actor)} -> {env_util.agent_id_to_name(agents, self.perceptor)} (weight: {self.weight})'


class TrustConf:
    def __init__(self, player_id, anticipation, steps):
        self.player_id = player_id
        self.anticipation = anticipation
        self.steps = steps


def activate_config_file(conf_name):
    global n_step
    global griddly_description
    global agents
    conf_dict = load_conf_dict(conf_name)
    n_step = conf_dict['NStep']
    griddly_description = conf_dict['GriddlyDescription']
    agents = agents_conf_from_dict(conf_dict['Agents'])
    set_health_perf_consistency_from_obj(conf_dict)


def load_conf_dict(conf_name):
    with open('game_conf.yaml', 'r') as f:
        return yaml.safe_load(f)[conf_name]


def agents_conf_from_dict(agent_conf_dict):
    agent_confs = []
    for agent_conf in agent_conf_dict:
        new_agent_conf = AgentConf(agent_conf['Name'], agent_conf['PlayerId'])
        new_agent_conf.policy = getattr(policies, agent_conf['Policy'])
        new_agent_conf.assumed_policy = getattr(policies, agent_conf['AssumedPolicy'])
        if 'EmpowermentPairs' in agent_conf:
            new_agent_conf.empowerment_pairs = empowerment_pairs_from_dict(agent_conf['EmpowermentPairs'])
        new_agent_conf.actions = agent_conf['Actions']
        if 'MaxHealth' in agent_conf:
            new_agent_conf.max_health = agent_conf['MaxHealth']
        if 'Trust' in agent_conf:
            new_agent_conf.trust = trust_conf_from_dict(agent_conf['Trust'])
        agent_confs.append(new_agent_conf)
    return agent_confs


def empowerment_pairs_from_dict(empowerment_pairs_dict):
    empowerment_pairs = []
    for empowerment_pair_conf in empowerment_pairs_dict:
        empowerment_pairs.append(EmpowermentpairConf(empowerment_pair_conf['Actor'],
                                                     empowerment_pair_conf['Perceptor'],
                                                     empowerment_pair_conf['Weight']))
    return empowerment_pairs


def trust_conf_from_dict(trust_conf_dict):
    trust_confs = []
    for trust_conf in trust_conf_dict:
        trust_confs.append(TrustConf(trust_conf['PlayerId'],
                                     trust_conf['Anticipation'],
                                     trust_conf['Steps']))
    return trust_confs


def set_health_perf_consistency_from_obj(conf_obj):
    if 'HealthPerformanceConsistency' in conf_obj:
        set_health_performance_consistency(conf_obj['HealthPerformanceConsistency'])


def set_verbose_calculation(verbose):
    global verbose_calculation
    verbose_calculation = verbose


def set_health_performance_consistency(consistency):
    global health_performance_consistency
    health_performance_consistency = consistency


def set_visualise_all(visualise):
    global visualise_all
    visualise_all = visualise
    

def set_n_step(n_step_):
    global n_step
    n_step = n_step_
