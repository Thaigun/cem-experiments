import random
import matplotlib.pyplot as plt
import json
import os
import copy
import numpy as np
from enum import Enum
from itertools import product
import play_back_tool
import action_space_builder
from ptitprince import PtitPrince as pt
import video_exporter
from datetime import datetime
import pandas as pd
from create_griddly_env import create_griddly_env


TEST_GROUP_SIZE = 30*4*3
save_folder = ''


class CollectActionType(Enum):
    SEPARATE = 0
    EMBEDDED = 1


class SubDataSet:
    def __init__(self, file_prefix, name, data):
        self.file_prefix = file_prefix
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
            

def select_with_collect_type(full_data, type):
    selected_runs = []
    for game_rules_obj in full_data['game_rules'].values():
        separate_collect = False
        if 'collect' in game_rules_obj['PlayerActions'] or 'collect_from_ahead' in game_rules_obj['PlayerActions']:
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


def plot_difference_histograms(data_set, save_folder):
    figure, axs = plt.subplots(3, 1)
    figure.set_tight_layout(True)
    data = data_set.data
    test_batches = group_runs_by_params(data, ['GriddlyDescription', 'GameRules', 'Map'])
    score_diffs_per_pair = extract_score_diffs_per_pair(data, test_batches)

    emp_param_names = get_cem_param_names(data)
    pair_order = [
        'Supportive-Antagonistic',
        'Random-Antagonistic',
        'Random-Supportive'
    ]

    for pair, diffs in score_diffs_per_pair.items():
        pair_name = emp_param_names[pair[1]] + '-' + emp_param_names[pair[0]]
        print(pair_name, 'Mean difference:', np.mean(diffs))
        sub_plot_idx = pair_order.index(pair_name)
        axs[sub_plot_idx].hist(diffs, bins=16, range=(-8.5, 7.5), linewidth=0.5, edgecolor='white')
        if sub_plot_idx == 0:
            axs[sub_plot_idx].set_title(data_set.name + ', ' + emp_param_names[pair[1]] + '-' + emp_param_names[pair[0]])
        if sub_plot_idx == len(pair_order) - 1:
            axs[sub_plot_idx].set_xlabel('Score difference between CEM-parametrizations')
        axs[sub_plot_idx].set_ylabel('Number of runs')
    figure.set_size_inches(5.5, 7.5)
    if save_folder:
        figure.savefig(os.path.join(save_folder, data_set.file_prefix + 'diff_histograms.png'))
        plt.close()
    else:
        plt.show()

        
def extract_score_diffs_per_pair(full_data, test_batches):
    cem_pairs = get_cem_param_pairs(full_data)
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


def plot_run_score_matrix(full_data, save_folder):
    def prepare_raincloud_data(full_data, from_runs):
        cem_names = get_cem_param_names(full_data)
        data = build_data_for_selected_runs(full_data, from_runs)
        groups_to_average = group_runs_by_params(data, ['CemParams', 'GameRules'])
        avg_scores_per_group = {group_key: np.mean([run['Score'] for run in group_runs]) for group_key, group_runs in groups_to_average.items()}
        df_data = []
        for group_key, avg_score in avg_scores_per_group.items():
            cem_name = cem_names[group_key[0]]
            df_data.append((cem_name, avg_score))
        return pd.DataFrame.from_records(df_data, columns=['cem_param', 'action_set_avg'])
    
    figure, axs = plt.subplots(1, len(full_data['map_params']))
    figure.set_tight_layout(True)
    cem_order = ['Supportive', 'Random', 'Antagonistic']
    runs_per_map = group_runs_by_params(full_data, ['MapParams'], return_keys=True)
    map_names = get_map_param_names(full_data)
    map_key_i = 0
    for map_param_key, runs in runs_per_map.items():
        data_frame = prepare_raincloud_data(full_data, runs)
        axs[map_key_i] = pt.RainCloud(x='cem_param', y='action_set_avg', data=data_frame, palette='Set2', ax=axs[map_key_i], orient='h', order=cem_order, bw=0.2)
        axs[map_key_i].set_title(map_names[map_param_key[0]])
        axs[map_key_i].xaxis.grid(visible=True)
        axs[map_key_i].yaxis.set_visible(map_key_i == 0)
        map_key_i += 1
    figure.set_size_inches(11, 6)
    if save_folder:
        figure.savefig(os.path.join(save_folder, 'avg_result_matrix.png'))
        plt.close()
    else:
        plt.show()


def plot_avg_diff_rainclouds(data_set, save_folder):
    def make_cem_param_name(cem_param_names, pair):
        short_names = {
            'Supportive': 'sup',
            'Antagonistic': 'ant',
            'Random': 'rnd'
        }
        return short_names[cem_param_names[pair[1]]] + '-' + short_names[cem_param_names[pair[0]]]

    def print_outliers(title, lo_outliers, hi_outliers):
        print('Outliers for pair', title)
        print('Low outliers:')
        for lo_outlier in lo_outliers:
            print('GameRules:', lo_outlier[0][0], 'MapParams:', lo_outlier[0][2], 'Value:', lo_outlier[1])
        print('High outliers:')
        for hi_outlier in hi_outliers:
            print('GameRules:', hi_outlier[0][0], 'MapParams:', hi_outlier[0][2], 'Value:', hi_outlier[1])

    figure, axs = plt.subplots()
    data = data_set.data
    cem_param_pairs = get_cem_param_pairs(data)
    avg_list_per_cem_pair = {pair: [] for pair in cem_param_pairs}
    game_variant_batches = group_runs_by_params(data, ['GameRules', 'GriddlyDescription', 'MapParams'])
    for game_variant_key, game_variant_batch in game_variant_batches.items():
        variant_runs_per_cem = {cem_param: [] for cem_param in data['cem_params'].keys()}
        for game_run in game_variant_batch:
            variant_runs_per_cem[game_run['CemParams']].append(game_run)
        avg_score_per_cem = {}
        for cem_param, runs in variant_runs_per_cem.items():
            avg_score = np.mean([run['Score'] for run in runs])
            avg_score_per_cem[cem_param] = avg_score
        for pair in cem_param_pairs:
            avg_diff = avg_score_per_cem[pair[1]] - avg_score_per_cem[pair[0]]
            avg_list_per_cem_pair[pair].append((game_variant_key, avg_diff))

    cem_param_names = get_cem_param_names(data)
    pd_ready_data = []
    for pair, data_points in avg_list_per_cem_pair.items():
        cem_pair_name = make_cem_param_name(cem_param_names, pair)
        low_outliers, hi_outliers = find_outliers(data_points)
        print_outliers(cem_pair_name, low_outliers, hi_outliers)
        pd_ready_data += [(cem_pair_name, data_point[1]) for data_point in data_points]
    data_frame = pd.DataFrame.from_records(pd_ready_data, columns=['cem_pair', 'score_diff_avg'])

    axs = pt.RainCloud(x='cem_pair', y='score_diff_avg', data=data_frame, palette='Set2', ax=axs, order=['sup-ant', 'rnd-ant', 'rnd-sup'], orient='h', bw=0.2)
    axs.xaxis.grid(visible=True)
    axs.set_title('Mean score difference between different CEM-parametrizations\n' + data_set.name)
    figure.set_size_inches(5.5, 7.5)
    if save_folder:
        figure.savefig(os.path.join(save_folder, data_set.file_prefix+'avg_diff_raincloud.png'))
        plt.close()
    else:
        plt.show()


def find_outliers(data_keys_and_values, outlier_const=1.5):
    data_arr = np.array([pair[1] for pair in data_keys_and_values])
    upper_quartile = np.percentile(data_arr, 75)
    lower_quartile = np.percentile(data_arr, 25)
    iqr = (upper_quartile - lower_quartile) * outlier_const
    outlier_bounds = (lower_quartile - iqr, upper_quartile + iqr)
    low_outliers = []
    high_outliers = []
    for data_key, data_val in data_keys_and_values:
        if data_val < outlier_bounds[0]:
            low_outliers.append((data_key, data_val))
        elif data_val > outlier_bounds[1]:
            high_outliers.append((data_key, data_val))
    return low_outliers, high_outliers


def plot_all_action_frequencies(data_set, save_folder):
    def get_action_name(env_, action):
        return 'idle' if action[1] == 0 else env_.action_names[action[0]]
        
    data = data_set.data
    env = create_griddly_env('collector_game.yaml')
    env.reset()
    cem_name_lookup = get_cem_param_names(data)

    labels = []
    cem_keys = list(data['cem_params'])
    cem_names = [cem_name_lookup[key] for key in cem_keys]
    data_per_agent = []
    for agent_i in range(env.player_count):
        data_per_agent.append([[] for _ in cem_keys])

    for game_run in data['game_runs'].values():
        agent_in_turn = 0
        cem_key = game_run['CemParams']
        cem_idx = cem_keys.index(cem_key)
        for full_action in game_run['Actions']:
            agent_action = full_action[agent_in_turn]
            action_name = get_action_name(env, agent_action)
            if action_name not in labels:
                labels.append(action_name)
                for agent_i in range(env.player_count):
                    for data_list in data_per_agent[agent_i]:
                        data_list.append(0)
            action_name_idx = labels.index(action_name)
            data_per_agent[agent_in_turn][cem_idx][action_name_idx] += 1
            agent_in_turn = (agent_in_turn + 1) % env.player_count

    for agent_i in range(len(data_per_agent)):
        for cem_i in range(len(data_per_agent[agent_i])):
            sorted_by_labels = [sorted_data for _, sorted_data in sorted(zip(labels, data_per_agent[agent_i][cem_i]), key=lambda x: x[0])]
            data_per_agent[agent_i][cem_i] = sorted_by_labels
    labels.sort()
    
    fig, ax = plot_grouped_bars('Frequency of Player actions with different CEM-parametrizations\n'+data_set.name, labels, data_per_agent[0], cem_names, 'Frequency', 'Action')
    if save_folder:
        fig.savefig(os.path.join(save_folder, data_set.file_prefix + 'plr_action_freq.png'))
        plt.close()
    else:
        plt.show()

    fig, ax = plot_grouped_bars('Frequency of NPC actions with different CEM-parametrizations\n'+data_set.name, labels, data_per_agent[1], cem_names, 'Frequency', 'Action')
    if save_folder:
        fig.savefig(os.path.join(save_folder, data_set.file_prefix + 'npc_action_freq.png'))
        plt.close()
    else:
        plt.show()


def plot_grouped_bars(title, labels, data_sets, legend_names, y_name, x_name):
    x = np.arange(len(labels))
    width = 0.25
    figure, ax = plt.subplots()
    for i in range(len(data_sets)):
        data_set = data_sets[i]
        legend_name = legend_names[i]
        x_positions = x - (len(legend_names) - 1) * width/2 + i * width
        ax.bar(x_positions, data_set, width, label=legend_name)
    ax.set_ylabel('Occurrences')
    ax.set_xticks(x, labels, rotation='vertical')
    ax.legend()
    ax.set_title(title)
    figure.set_tight_layout(True)
    return figure, ax


def get_cem_param_pairs(full_data):
    cem_keys = list(full_data['cem_params'].keys())
    cem_pairs = {tuple(sorted(pair)) for pair in product(cem_keys, repeat=2) if pair[0] != pair[1]}
    return cem_pairs


def group_runs_by_params(data, param_names, *, return_keys=False):
    sorted_params = sorted(param_names)
    param_groups = {}
    for run_key, run in data['game_runs'].items():
        param_values = [run[param_name] for param_name in sorted_params]
        param_group_key = tuple(param_values)
        if param_group_key not in param_groups:
            param_groups[param_group_key] = []
        if return_keys:
            param_groups[param_group_key].append(run_key)
        else:
            param_groups[param_group_key].append(run)
    return param_groups


def get_cem_param_names(full_data):
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


def get_map_param_names(full_data):
    names = {}
    for map_param_key, map_param in full_data['map_params'].items():
        size = 'Big' if map_param['Width'] > 12 else 'Small'
        density = 'Dense' if map_param['ObjectCounts']['w']/(map_param['Width']*map_param['Height']) > 0.2 else 'Sparse'
        names[map_param_key] = size + ' ' + density
    return names


def create_data_sets(full_data):
    data_sets = []

    separate_collect_runs = select_with_collect_type(full_data, CollectActionType.SEPARATE)
    embedded_collect_runs = select_with_collect_type(full_data, CollectActionType.EMBEDDED)
    small_map_runs = select_with_map_size(full_data, 8, 8)
    big_map_runs = select_with_map_size(full_data, 14, 14)
    small_and_separate_runs = set(separate_collect_runs).intersection(set(small_map_runs))

    data_sets.append(SubDataSet('allruns_', 'All test runs', full_data))
    data_sets.append(SubDataSet('separatecollect_', 'Player with separate collect action', build_data_for_selected_runs(full_data, separate_collect_runs)))
    data_sets.append(SubDataSet('embeddedcollect_', 'Player\'s collect action embedded to movement', build_data_for_selected_runs(full_data, embedded_collect_runs)))
    data_sets.append(SubDataSet('smallmaps_', 'Small maps', build_data_for_selected_runs(full_data, small_map_runs)))
    data_sets.append(SubDataSet('bigmaps_', 'Big maps', build_data_for_selected_runs(full_data, big_map_runs)))
    data_sets.append(SubDataSet('smallmapsseparatecollect_', 'Small maps and separate collect action', build_data_for_selected_runs(full_data, small_and_separate_runs)))
    return data_sets


def do_plotting(full_data):
    save_folder = input('Enter folder to save plots to (empty for no save): ')
    if save_folder and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    test_data_sets = create_data_sets(full_data)
    for sub_data in test_data_sets:
        plot_all_action_frequencies(sub_data, save_folder)
        plot_difference_histograms(sub_data, save_folder)
        plot_avg_diff_rainclouds(sub_data, save_folder)
    plot_run_score_matrix(full_data, save_folder)


def do_zero_score_search(full_data):
    zero_score_games = select_with_run_score(full_data, 0, 0)
    print('Number of zero-score games: ', len(zero_score_games))
    print(zero_score_games)
    run_zero_scores = input('Do you want to replay zero score games? (y/n)')
    if run_zero_scores == 'y':
        for run_key in zero_score_games:
            play_back_tool.run_replay_for_id(run_key, delay=0.3)


def do_save_video_replays(full_data):
    game_rules_key = input('Enter game rules key (empty for random): ')
    map_params_key = input('Enter map params key (empty for random): ')
    n = int(input('Enter number of replay groups to save: '))
    game_run_groups = group_runs_by_params(full_data, ['GameRules', 'Map', 'MapParams'])

    filtered_group_keys = list(game_run_groups)
    if game_rules_key:
        filtered_group_keys = [key for key in filtered_group_keys if key[0] == game_rules_key]
    if map_params_key:
        filtered_group_keys = [key for key in filtered_group_keys if key[2] == map_params_key]

    selected_group_keys = random.sample(filtered_group_keys, n)
    cem_param_names = get_cem_param_names(full_data)
    sub_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    for group_key in selected_group_keys:
        game_rules = full_data['game_rules'][group_key[0]]
        map_hash = str(hash(group_key[1]))[:5]
        sub_sub_dir = os.path.join(sub_dir, '-'.join(game_rules['PlayerActions']) + '__' + '-'.join(game_rules['NpcActions']) + map_hash)
        for game_run in game_run_groups[group_key]:
            cem_param_name = cem_param_names[game_run['CemParams']]
            video_exporter.make_video_from_data(game_run, sub_sub_dir, cem_param_name, 40)


if __name__ == '__main__':
    result_obj = get_result_object()
    complete_test_group_data = select_complete_test_groups(result_obj)
    
    plotting = input('Plot figures? (y/n) ')
    if plotting == 'y':
        do_plotting(complete_test_group_data)

    find_zeroes = input('Find runs with score 0? (y/n) ')
    if find_zeroes == 'y':
        do_zero_score_search(complete_test_group_data)

    video_output = input('Do you want to create a video of runs? (y/n) ')
    if video_output == 'y':
        do_save_video_replays(complete_test_group_data)
