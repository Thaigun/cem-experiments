import matplotlib.pyplot as plt
import json
import os
import copy
import numpy as np
from enum import Enum
from itertools import product
import play_back_tool
import action_space_builder


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
    complete_game_rules = []
    for game_rule_obj in all_game_rules.values():
        test_group_finished = 'game_runs' in game_rule_obj and len(game_rule_obj['game_runs']) == TEST_GROUP_SIZE
        action_builder = action_space_builder.CollectorActionSpaceBuilder()
        valid_action_set = action_builder.player_set_validation(set(game_rule_obj['PlayerActions']))
        if test_group_finished and valid_action_set:
            complete_game_rules.append(game_rule_obj)
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

    build_top_level_obj(full_data, run_keys, filtered_result, 'game_rules')
    build_top_level_obj(full_data, run_keys, filtered_result, 'cem_params')
    build_top_level_obj(full_data, run_keys, filtered_result, 'map_params')
    return filtered_result

def build_top_level_obj(full_data, run_keys, current_data_obj, top_level_name):
    for param_obj_key, param_obj in full_data[top_level_name].items():
        if not 'game_runs' in param_obj:
            continue
        filtered_param_obj = copy.deepcopy(param_obj)
        for dict_key, game_run_key in param_obj['game_runs'].items():
            if game_run_key not in run_keys:
                del filtered_param_obj['game_runs'][dict_key]
        if len(filtered_param_obj['game_runs']) > 0:
            current_data_obj[top_level_name][param_obj_key] = filtered_param_obj


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
    figure, axs = plt.subplots(3, 3)
    plot_diff_histograms(axs[0], data_set)
    #plot_avg_diff_raincloud(data)
    #plot_proportion_bars(data)
    plt.show()

def plot_diff_histograms(ax, data_set):
    data = data_set.data
    score_diffs_per_pair = extract_score_diffs_per_pair(data)

    emp_param_names = get_emp_param_names(data)
    sub_plot_idx = 0
    for pair, diffs in score_diffs_per_pair.items():
        print(emp_param_names[pair[1]], '-', emp_param_names[pair[0]])
        print(np.mean(diffs))
        ax[sub_plot_idx].hist(diffs, bins=16, range=(-8.5, 7.5), linewidth=0.5, edgecolor='white')
        ax[sub_plot_idx].set_title(data_set.name + ',\n' + emp_param_names[pair[1]] + ' - ' + emp_param_names[pair[0]])
        ax[sub_plot_idx].set_xlabel('Score difference between CEM-parametrizations')
        ax[sub_plot_idx].set_ylabel('Number of runs')
        sub_plot_idx += 1

def extract_score_diffs_per_pair(data):
    test_batches = extract_test_patches(data)

    cem_keys = list(data['cem_params'].keys())
    cem_pairs = {tuple(sorted(pair)) for pair in product(cem_keys, repeat=2) if pair[0] != pair[1]}
    score_diffs_per_pair = {pair: [] for pair in cem_pairs}
    for test_batch in test_batches.values():
        for run1_idx in range(len(test_batch)):
            for run2_idx in range(run1_idx + 1, len(test_batch)):
                run1 = test_batch[run1_idx]
                run2 = test_batch[run2_idx]
                if run1 == run2:
                    continue
                cem_pair = tuple(sorted([run1['CemParams'], run2['CemParams']]))
                diff = run2['Score'][0] - run1['Score'][0]
                # Make sure the diff is "cem_pair_1"-"cem_pair_0" and not "cem_pair_0"-"cem_pair_1"
                if run2['CemParams'] == cem_pair[0]:
                    diff = -diff
                score_diffs_per_pair[cem_pair].append(diff)
    return score_diffs_per_pair

def extract_test_patches(data):
    test_batches = {}
    for run_key in data['game_runs']:
        game_run = data['game_runs'][run_key]
        test_batch_key = (game_run['GriddlyDescription'], game_run['GameRules'], game_run['Map'])
        if test_batch_key not in test_batches:
            test_batches[test_batch_key] = []
        test_batches[test_batch_key].append(game_run)
    return test_batches


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


def create_data_sets(full_data):
    data_sets = []

    separate_collect_runs = select_with_collect_type(full_data, CollectActionType.SEPARATE)
    embedded_collect_runs = select_with_collect_type(full_data, CollectActionType.EMBEDDED)
    small_map_runs = select_with_map_size(full_data, 8, 8)
    big_map_runs = select_with_map_size(full_data, 14, 14)
    small_and_separate_runs = set(separate_collect_runs).intersection(set(small_map_runs))

    data_sets.append(SubDataSet('All test runs', full_data))
    data_sets.append(SubDataSet('Player with separate collect action', build_data_for_selected_runs(full_data, separate_collect_runs)))
    data_sets.append(SubDataSet('Player\'s collect action embedded to movement', build_data_for_selected_runs(full_data, separate_collect_runs)))
    data_sets.append(SubDataSet('Small maps', build_data_for_selected_runs(full_data, small_map_runs)))
    data_sets.append(SubDataSet('Big maps', build_data_for_selected_runs(full_data, big_map_runs)))
    data_sets.append(SubDataSet('Small maps and separate collect action', build_data_for_selected_runs(full_data, small_and_separate_runs)))

    return data_sets
        

if __name__ == '__main__':
    result_obj = get_result_object()
    complete_test_group_data = select_complete_test_groups(result_obj)
    test_data_sets = create_data_sets(complete_test_group_data)

    for sub_data in test_data_sets:
        plot_all_figures(sub_data)
    zero_score_games = select_with_run_score(complete_test_group_data, 0, 0)
    print('Number of zero-score games: ', len(zero_score_games))
    print(zero_score_games)
    run_zero_scores = input('Do you want to replay zero score games? (y/n)')
    if run_zero_scores == 'y':
        for run_key in zero_score_games:
            play_back_tool.run_replay_for_id(run_key, delay=0.3)
