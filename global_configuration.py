import yaml

verbose_calculation = False
health_performance_consistency = True
visualise_all = False
time_limit = 2


def activate_config_file(conf_name):
    global n_step
    global griddly_description
    global agents
    conf_dict = load_conf_dict(conf_name)
    set_health_perf_consistency_from_obj(conf_dict)
    if 'MCTSTimeLimit' in conf_dict:
        set_time_limit(conf_dict['MCTSTimeLimit'])


def load_conf_dict(conf_name):
    with open('game_conf.yaml', 'r') as f:
        return yaml.safe_load(f)[conf_name]


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


def set_time_limit(time_limit_):
    global time_limit
    time_limit = time_limit_
