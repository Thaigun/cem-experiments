import matplotlib.pyplot as plt
import json
import os
import copy
import numpy as np


TEST_GROUP_SIZE = 30*4*3


def get_result_object():
    result_file_name = 'prod5.json' #input('Enter the name of the result file: ')
    with open(os.path.join('results', result_file_name), 'r') as result_file:
        result_object = json.load(result_file)
        return result_object


def select_complete_test_groups(result_obj):
    filtered_result = {
        'cem_params': {},
        'map_params': {},
        'game_rules': {},
        'game_runs': {}
    }

    all_game_rules = result_obj['game_rules']
    complete_game_rule_keys = [game_rule_key for game_rule_key, game_rule_obj in all_game_rules.items() if len(game_rule_obj['game_runs']) == TEST_GROUP_SIZE]
    for game_rule_key in complete_game_rule_keys:
        filtered_result['game_rules'][game_rule_key] = copy.deepcopy(all_game_rules[game_rule_key])

    game_runs_in_complete_groups = [game_run_key for game_run_key, game_run_obj in result_obj['game_runs'].items() if game_run_obj['GameRules'] in complete_game_rule_keys]
    for game_run_key in game_runs_in_complete_groups:
        filtered_result['game_runs'][game_run_key] = result_obj['game_runs'][game_run_key] #copy.deepcopy(result_obj['game_runs'][game_run_key])

    for cem_param_key, cem_param_obj in result_obj['cem_params'].items():
        filtered_cem_param = copy.deepcopy(cem_param_obj)
        for dict_key, game_run_key in cem_param_obj['game_runs'].items():
            if game_run_key not in game_runs_in_complete_groups:
                del filtered_cem_param['game_runs'][dict_key]
        filtered_result['cem_params'][cem_param_key] = filtered_cem_param

    for map_param_key, map_param_obj in result_obj['map_params'].items():
        filtered_map_param = copy.deepcopy(map_param_obj)
        for dict_key, game_run_key in map_param_obj['game_runs'].items():
            if game_run_key not in game_runs_in_complete_groups:
                del filtered_map_param['game_runs'][dict_key]
        filtered_result['map_params'][map_param_key] = filtered_map_param

    return filtered_result


def boxplot_cem_comparison(result_obj):
    cem_scores = []
    for cem_param_obj in result_obj['cem_params'].values():
        print(cem_param_obj['EmpowermentPairs'])
        this_cem_scores = []
        for game_run_key in cem_param_obj['game_runs'].values():
            this_cem_scores.append(result_obj['game_runs'][game_run_key]['Score'][0])
        np_scores = np.array(this_cem_scores)
        print('Average: ', np.mean(np_scores))
        cem_scores.append(np_scores)
    plot_data(cem_scores, 'CEM-parametrization comparison')


def boxplot_separate_collect_comparison(result_obj):
    separate_collect_runs = []
    embedded_collect_runs = []
    for game_rules_obj in result_obj['game_rules'].values():
        separate_collect = False
        if 'collect' in game_rules_obj['PlayerActions'] or 'collect_ahead' in game_rules_obj['PlayerActions']:
            separate_collect = True
        if separate_collect:
            separate_collect_runs += game_rules_obj['game_runs'].values()
        else:
            embedded_collect_runs += game_rules_obj['game_runs'].values()

    separate_cem_scores = []
    embedded_cem_scores = []
    for cem_param_obj in result_obj['cem_params'].values():
        this_separate_cem_scores = []
        this_embedded_cem_scores = []
        for game_run_key in cem_param_obj['game_runs'].values():
            if game_run_key in separate_collect_runs:
                this_separate_cem_scores.append(result_obj['game_runs'][game_run_key]['Score'][0])
            else:
                this_embedded_cem_scores.append(result_obj['game_runs'][game_run_key]['Score'][0])
        separate_np_scores = np.array(this_separate_cem_scores)
        embedded_np_scores = np.array(this_embedded_cem_scores)
        print('Average separate: ', np.mean(separate_np_scores))
        print('Average embedded: ', np.mean(embedded_np_scores))
        separate_cem_scores.append(separate_np_scores)
        embedded_cem_scores.append(embedded_np_scores)
    plot_data(separate_cem_scores, 'CEM-parametrization comparison')
    plot_data(embedded_cem_scores, 'CEM-parametrization comparison')


def plot_data(data, name):
    fig1, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.boxplot(data)
    plt.show()


if __name__ == '__main__':
    result_obj = get_result_object()
    complete_test_group_data = select_complete_test_groups(result_obj)
    boxplot_cem_comparison(complete_test_group_data)
    boxplot_separate_collect_comparison(complete_test_group_data)
