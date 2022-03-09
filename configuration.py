import yaml
import policies

active_config = None
verbose_calculation = False
health_performance_consistency = True
visualise_all = False


def activate_config(conf_name):
    global active_config
    active_config = load_pure_conf(conf_name)
    replace_policies_with_functions(active_config)
    set_health_perf_consistency_from_obj(active_config)


def load_pure_conf(conf_name):
    with open('game_conf.yaml', 'r') as f:
        return yaml.safe_load(f)[conf_name]


def replace_policies_with_functions(conf_object):
     for agent_conf in conf_object['Agents']:
        agent_conf['AssumedPolicy'] = getattr(policies, agent_conf['AssumedPolicy'])
        # Agents that are controlled by a human player are special cases and are skipped
        if agent_conf['Policy'] != 'KBM':
            agent_conf['Policy'] = getattr(policies, agent_conf['Policy'])


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
    