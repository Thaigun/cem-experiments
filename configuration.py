import yaml
import policies

active_config = None
verbose_calculation = False
health_performance_consistency = True

def activate_config(conf_name):
    global active_config
    with open('game_conf.yaml', 'r') as f:
        active_config = yaml.safe_load(f)[conf_name]

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
