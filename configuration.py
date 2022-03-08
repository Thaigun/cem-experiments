import yaml
import policies

active_config = None
active_config_name = ""
verbose_calculation = False
health_performance_consistency = True
visualise_all = False


def load_pure_conf():
    global active_config_name
    with open('game_conf.yaml', 'r') as f:
        return yaml.safe_load(f)[active_config_name]


def activate_config(conf_name):
    global active_config_name
    global active_config

    active_config_name = conf_name
    active_config = load_pure_conf()

    # Replace the policy values with functions of the same name
    for agent_conf in active_config['Agents']:
        agent_conf['AssumedPolicy'] = getattr(policies, agent_conf['AssumedPolicy'])
        # Agents that are controlled by a human player are special cases
        if agent_conf['Policy'] != 'KBM':
            agent_conf['Policy'] = getattr(policies, agent_conf['Policy'])

    if 'HealthPerformanceConsistency' in active_config:
        global health_performance_consistency
        health_performance_consistency = active_config['HealthPerformanceConsistency']


def set_verbose_calculation(verbose):
    global verbose_calculation
    verbose_calculation = verbose


def set_health_performance_consistency(consistency):
    global health_performance_consistency
    health_performance_consistency = consistency


def set_visualise_all(visualise):
    global visualise_all
    visualise_all = visualise
    