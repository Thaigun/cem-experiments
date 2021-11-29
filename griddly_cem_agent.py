from itertools import product
import empowerment_maximization
import numpy as np

# Implement a simple function that hashes the numpy array and returns the hash
def hash_obs(obs):
    return hash(str(obs))


def build_action(action, player_id, player_count):
    built_action = [[0,0] for _ in range(player_count)]
    built_action[player_id-1] = list(action)
    return built_action


def build_mapping(env, current_hash, mapping, steps_left, action_space, current_player, main_player, samples=1):
    if steps_left == 0 or current_hash in mapping:
        return
    
    mapping[current_hash] = [{} for _ in action_space]

    for action_idx, action in enumerate(action_space):
        cpd_s_a = {}
        sample_weight = 1.0/samples
        # Take the action several times to sample the probabilities of future states
        for _ in range(samples):
            clone_env = env.clone()
            clone_env.step(build_action(action, current_player, env.player_count))
            state_after_hash = clone_env.get_state()['Hash']
            if state_after_hash not in cpd_s_a:
                cpd_s_a[state_after_hash] = sample_weight
            else:
                cpd_s_a[state_after_hash] += sample_weight
            build_mapping(clone_env, state_after_hash, mapping, steps_left-1, action_space, current_player % env.player_count + 1, main_player, samples)
        mapping[current_hash][action_idx] = cpd_s_a


def build_distribution(env, actions, action_stepper, action_space, for_player, current_step_player, steps, hash_decode, samples):
    if steps == 0:
        hash = env.get_state()['Hash']
        if hash not in hash_decode:
            hash_decode[hash] = env
        return {hash: 1.0}
    
    actions = [actions[action_stepper]] if for_player == current_step_player else [a for a in action_space]
    next_probs = {}
    coeff = 1 / len(actions)
    for action_idx, action in enumerate(actions):
        clone_env = env.clone()
        obs, rew, env_done, info = clone_env.step(build_action(action, current_step_player, env.player_count))
        next_distribution = build_distribution(clone_env, actions, action_stepper + 1, action_space, for_player, current_step_player % env.player_count + 1, steps-1, hash_decode, samples)
        for key in next_distribution:
            if key not in next_probs:
                next_probs[key] = 0
            next_probs[key] += next_distribution[key] * coeff
    return next_probs


def cem_action(env, current_player, steps, empowerment_pairs, empowerment_weights, samples=1):
    '''
    env: the game environment
    player_id: the id of the player to take an action for
    steps: the number of steps (n-step empowerment)
    empowerment_pairs: an array of tuples, where the first element is the playerId whos actuator 
        is the input and the second element is the playerId whose perceptor is the output of the communication channel
    empowerment_weights: an array of floats, where the ith element is the weight of the ith empowerment pair to be used 
        in the action policy
    '''
    action_space = [(0, 0)] # Include the idling action
    for action_type_index, action_name in enumerate(env.action_names):
        for action_id in range(1, env.num_action_ids[action_name]):
            action_space.append((action_type_index, action_id))

    expected_empowerments_per_pair = []

    for emp_pair in empowerment_pairs:
        expected_empowerments = np.zeros(len(action_space))
        hash_decode = {}
        anticipation_step_count = emp_pair[0] - current_player if emp_pair[0] > current_player else env.player_count - (current_player - emp_pair[0])
        anticipation = [None] * len(action_space)
        for action_idx, action in enumerate(action_space):
            anticipation[action_idx] = build_distribution(env, [action], 0, action_space, current_player, current_player, anticipation_step_count, hash_decode, samples)
        # Calculate the n-step empowerment for each state that was found earlier
        reachable_states =[]
        for a in anticipation:
            for key in a:
                reachable_states.append(key)
        # All possible action combinations of length 'step'
        action_combinations = [list(combo) for combo in product(action_space, repeat=steps)]
        nstep_combo_probs = [None] * len(action_combinations)
        state_empowerments = {}
        for state in reachable_states:
            for combo_idx, action_combo in enumerate(action_combinations):
                nstep_combo_probs[combo_idx] = build_distribution(hash_decode[state], action_combo, 0, action_space, emp_pair[0], emp_pair[0], steps * env.player_count, hash_decode, samples)
            
            cpd_s_a = [{} for _ in nstep_combo_probs]
            for c_id, c in enumerate(nstep_combo_probs):
                for key in c:
                    latest_obs = hash_decode[key]._player_last_observation[emp_pair[1]-1]
                    hashed_obs = hash_obs(latest_obs)
                    if hashed_obs not in cpd_s_a[c_id]:
                        cpd_s_a[c_id][hashed_obs] = c[key]
                    else:
                        cpd_s_a[c_id][hashed_obs] += c[key]
            # Use the Blahut-Arimoto algorithm to calculate the optimal probability distribution
            pd_a_opt = empowerment_maximization.blahut_arimoto(cpd_s_a, np.random.default_rng())
            # Use the optimal distribution to calculate the mutual information (empowerement)
            empowerment = empowerment_maximization.mutual_information(pd_a_opt, cpd_s_a)
            state_empowerments[state] = empowerment

        # Calculate the expected empowerment for each action that can be taken here
        for a in range(0, len(action_space)):
            for s, p_s in anticipation[a].items():
                expected_empowerments[a] += p_s * state_empowerments[s]
        expected_empowerments_per_pair.append(expected_empowerments)

    overall_rewards = []
    # Find the action index that yields th highest expected empowerment
    for a in range(len(action_space)):
        policy_reward = 0
        for e in range(len(empowerment_pairs)):
            policy_reward += empowerment_weights[e] * expected_empowerments_per_pair[e][a]
        overall_rewards.append(policy_reward)

    best_action_idx = np.argmax(overall_rewards)
    return action_space[best_action_idx]

