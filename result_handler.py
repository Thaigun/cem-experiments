import matplotlib.pyplot as plt
import json
import os
import copy
import numpy as np
from enum import Enum
from itertools import product


TEST_GROUP_SIZE = 30*4*3


class CollectActionType(Enum):
    SEPARATE = 0
    EMBEDDED = 1


class SubDataSet:
    def __init__(self, name, data):
        self.name = name
        self.data = data


def get_result_object():
    result_file_name = input('Enter the name of the result file: ')
    with open(os.path.join('results', result_file_name), 'r') as result_file:
        result_object = json.load(result_file)
        return result_object


def select_complete_test_groups(result_obj):
    all_game_rules = result_obj['game_rules']
    complete_game_rules = [game_rule_obj for game_rule_obj in all_game_rules.values() if 'game_runs' in game_rule_obj and len(game_rule_obj['game_runs']) == TEST_GROUP_SIZE]
    complete_test_run_keys = []
    for game_rule in complete_game_rules:
        complete_test_run_keys += game_rule['game_runs'].values()
    return build_data_for_selected_runs(result_obj, complete_test_run_keys)


def build_data_for_selected_runs(full_data, run_keys):
    filtered_result = {
        'cem_params': {},
        'map_params': {},
        'game_rules': {},
        'game_runs': {}
    }
    
    for game_run_key in run_keys:
        filtered_result['game_runs'][game_run_key] = full_data['game_runs'][game_run_key] #copy.deepcopy(result_obj['game_runs'][game_run_key])

    for game_rule_key, game_rule_obj in full_data['game_rules'].items():
        filtered_rule_param = copy.deepcopy(game_rule_obj)
        for dict_key, game_rule_key in game_rule_obj['game_runs'].items():
            if game_run_key not in run_keys:
                del filtered_rule_param['game_runs'][dict_key]
        filtered_result['game_rules'][game_rule_key] = filtered_rule_param

    for cem_param_key, cem_param_obj in full_data['cem_params'].items():
        filtered_cem_param = copy.deepcopy(cem_param_obj)
        for dict_key, game_run_key in cem_param_obj['game_runs'].items():
            if game_run_key not in run_keys:
                del filtered_cem_param['game_runs'][dict_key]
        filtered_result['cem_params'][cem_param_key] = filtered_cem_param

    for map_param_key, map_param_obj in full_data['map_params'].items():
        filtered_map_param = copy.deepcopy(map_param_obj)
        for dict_key, game_run_key in map_param_obj['game_runs'].items():
            if game_run_key not in run_keys:
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
    box_plot_data(cem_scores, 'CEM-parametrization comparison')


def box_plot_data(data, name):
    fig1, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.boxplot(data)
    plt.show()


def select_with_collect_type(full_data, type):
    selected_runs = []
    for game_rules_obj in full_data['game_rules'].values():
        separate_collect = False
        if 'collect' in game_rules_obj['PlayerActions'] or 'collect_ahead' in game_rules_obj['PlayerActions']:
            separate_collect = True
        if separate_collect and type is CollectActionType.SEPARATE or not separate_collect and type is CollectActionType.EMBEDDED:
            selected_runs += [game_run_key for game_run_key in game_rules_obj['game_runs'].values()]
    return selected_runs


def select_with_map_size(full_data, map_width, map_height):
    selected_runs = []
    for map_params in full_data['map_params'].values():
        if map_params['Height'] == map_height and map_params['Width'] == map_width:
            selected_runs += [game_run_key for game_run_key in map_params['game_runs'].values()]
    return selected_runs


def select_with_run_score(full_data, min_score, max_score):
    selected_runs = []
    for game_run_key, game_run_obj in full_data['game_runs'].items():
        if game_run_obj['Score'][0] >= min_score and game_run_obj['Score'][0] <= max_score:
            selected_runs.append(game_run_key)
    return selected_runs


def plot_all_figures(data_set):
    figure, axis = plt.subplots(3, 3)
    plot_diff_histogram(data_set, axis[0])
    #plot_avg_diff_raincloud(data)
    #plot_proportion_bars(data)
    plt.show()

def plot_diff_histogram(data_set, subplot):
    data = data_set.data
    cem_keys = list(data['cem_params'].keys())

    all_runs = {}
    for run_key in data['game_runs']:
        game_run = data['game_runs'][run_key]
        comparison_key = (game_run['GriddlyDescription'], game_run['GameRules'], game_run['Map'])
        if comparison_key not in all_runs:
            all_runs[comparison_key] = []
        all_runs[comparison_key].append(game_run)

    cem_pairs = {tuple(sorted(pair)) for pair in product(cem_keys, repeat=2) if pair[0] != pair[1]}
    diff_per_pair = {pair: [] for pair in cem_pairs}
    for key, group in all_runs.items():
        group_runs = group
        for run1_idx in range(len(group_runs)):
            for run2_idx in range(run1_idx + 1, len(group_runs)):
                run1 = group_runs[run1_idx]
                run2 = group_runs[run2_idx]
                if run1 == run2:
                    continue
                cem_pair = tuple(sorted([run1['CemParams'], run2['CemParams']]))
                diff = run2['Score'][0] - run1['Score'][0]
                if run2['CemParams'] == cem_pair[0]:
                    diff = -diff
                diff_per_pair[cem_pair].append(diff)

    emp_param_names = get_emp_param_names(data)
    sub_plot_idx = 0
    for pair, diffs in diff_per_pair.items():
        print(emp_param_names[pair[1]], '-', emp_param_names[pair[0]])
        print(np.mean(diffs))
        subplot[sub_plot_idx].hist(diffs, bins=16, range=(-8.5, 7.5))
        sub_plot_idx += 1


def get_emp_param_names(full_data):
    names = {}
    for cem_param_key, cem_param in full_data['cem_params'].items():
        if cem_param['Trust']['Anticipation']:
            names[cem_param_key] = 'Supportive'
            continue
        emp_pairs = cem_param['EmpowermentPairs']

        is_antagonistic = False
        for emp_pair in emp_pairs:
            if emp_pair['Actor'] == 1 and emp_pair['Perceptor'] == 1 and emp_pair['Weight'] < 0:
                is_antagonistic = True
                break
        if is_antagonistic:
            names[cem_param_key] = 'Antagonistic'
        else:
            names[cem_param_key] = 'Random'
    return names
        

if __name__ == '__main__':
    result_obj = get_result_object()
    complete_test_group_data = select_complete_test_groups(result_obj)

    test_data_sets = []

    all_test_runs = list(complete_test_group_data['game_runs'])
    separate_collect_runs = select_with_collect_type(complete_test_group_data, CollectActionType.SEPARATE)
    embedded_collect_runs = select_with_collect_type(complete_test_group_data, CollectActionType.EMBEDDED)
    small_map_runs = select_with_map_size(complete_test_group_data, 8, 8)
    big_map_runs = select_with_map_size(complete_test_group_data, 14, 14)
    small_and_separate_runs = set(separate_collect_runs).intersection(set(small_map_runs))

    test_data_sets.append(SubDataSet('All test runs', complete_test_group_data))
    test_data_sets.append(SubDataSet('Player with separate collect action', build_data_for_selected_runs(complete_test_group_data, separate_collect_runs)))
    test_data_sets.append(SubDataSet('Player\'s collect action embedded to movement', build_data_for_selected_runs(complete_test_group_data, separate_collect_runs)))
    test_data_sets.append(SubDataSet('Small maps', build_data_for_selected_runs(complete_test_group_data, small_map_runs)))
    test_data_sets.append(SubDataSet('Big maps', build_data_for_selected_runs(complete_test_group_data, big_map_runs)))
    test_data_sets.append(SubDataSet('Small maps and separate collect action', build_data_for_selected_runs(complete_test_group_data, small_and_separate_runs)))

    for sub_data in test_data_sets:
        plot_all_figures(sub_data)
    zero_score_games = select_with_run_score(complete_test_group_data, 0, 0)
    print('Number of zero-score games: ', len(zero_score_games))
    print(zero_score_games)
