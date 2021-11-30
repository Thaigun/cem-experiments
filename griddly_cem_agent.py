from itertools import product
import empowerment_maximization
import numpy as np

# Implement a simple function that hashes the numpy array and returns the hash
def hash_obs(obs):
    return hash(obs.tobytes())


def build_action(action, player_id, player_count):
    built_action = [[0,0] for _ in range(player_count)]
    built_action[player_id-1] = list(action)
    return built_action


def build_mapping(env, hash_decode, current_hash, mapping, steps_left, action_space, current_player, main_player, samples=1):
    if  current_hash in mapping:
        return

    if current_hash not in hash_decode:
        hash_decode[current_hash] = env

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
            build_mapping(clone_env, hash_decode, state_after_hash, mapping, steps_left-1, action_space, current_player % env.player_count + 1, main_player, samples)
        mapping[current_hash][action_idx] = cpd_s_a


def build_distribution(env_hash, mapping, actions, action_stepper, action_space, for_player, current_step_player, player_count, steps):
    if steps == 0:
        return {env_hash: 1.0}
    
    # If this step is for the player whose empowerment is being calculated, take the next action from the actions list. Otherwise, we use all possible actions 0 ... len(action_space-1)
    step_actions = [actions[action_stepper]] if for_player == current_step_player else range(len(action_space))
    next_probs = {}
    coeff = 1 / len(step_actions)
    for action in step_actions:
        next_hash = mapping[env_hash][action]
        for hash_prob in next_hash.items():
            next_action_step = action_stepper + 1 if for_player == current_step_player else action_stepper
            next_distribution = build_distribution(hash_prob[0], mapping, actions, next_action_step, action_space, for_player, current_step_player % player_count + 1, player_count, steps-1)
            for key in next_distribution:
                if key not in next_probs:
                    next_probs[key] = 0
                next_probs[key] += next_distribution[key] * coeff * hash_prob[1]
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

    anticipation_step_counts = [(emp_pair[0] - current_player if emp_pair[0] > current_player else env.player_count - (current_player - emp_pair[0])) for emp_pair in empowerment_pairs]
    mapping = {}
    hash_decode = {}
    build_mapping(env, hash_decode, env.get_state()['Hash'], mapping, max(anticipation_step_counts) + steps * env.player_count, action_space, current_player, current_player, samples)

    for pair_i, emp_pair in enumerate(empowerment_pairs):
        expected_empowerments = np.zeros(len(action_space))
        anticipation_step_count = anticipation_step_counts[pair_i]
        anticipation = [None] * len(action_space)
        for action_idx, action in enumerate(action_space):
            anticipation[action_idx] = build_distribution(env.get_state()['Hash'], mapping, [action_idx], 0, action_space, current_player, current_player, env.player_count, anticipation_step_count)
        # Calculate the n-step empowerment for each state that was found earlier
        reachable_states =[]
        for a in anticipation:
            for key in a:
                reachable_states.append(key)
        # All possible action combinations of length 'step'
        action_combinations = [list(combo) for combo in product(range(len(action_space)), repeat=steps)]
        nstep_combo_probs = [None] * len(action_combinations)
        state_empowerments = {}
        for state in reachable_states:
            for combo_idx, action_combo in enumerate(action_combinations):
                nstep_combo_probs[combo_idx] = build_distribution(state, mapping, action_combo, 0, action_space, emp_pair[0], emp_pair[0], env.player_count, steps * env.player_count)
            
            cpd_s_a = [{} for _ in nstep_combo_probs]
            for c_id, c in enumerate(nstep_combo_probs):
                for key in c:
                    latest_obs = hash_decode[key]._player_last_observation[emp_pair[1]-1]
                    hashed_obs = hash_obs(latest_obs)
                    if hashed_obs not in cpd_s_a[c_id]:
                        cpd_s_a[c_id][hashed_obs] = 0
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

