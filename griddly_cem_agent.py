from itertools import product
import empowerment_maximization
import numpy as np
import queue
import random

# Hashes the numpy array of observations
def hash_obs(obs):
    return hash(obs.tobytes())


# Builds the action that can be passed to the Griddly environment
def build_action(action, player_id, player_count):
    built_action = [[0,0] for _ in range(player_count)]
    built_action[player_id-1] = list(action)
    return built_action


# Do a breadth first search on the env to build the mapping from each state to the next states with probabilities. The mapping covers all states that are reachable in 'steps' steps.
def build_mapping(env, hash_decode, steps, action_space, current_player, samples=1):
    mapping = {}
    state_q = queue.Queue()

    # Initialize the breadth-first search
    init_state_hash = env.get_state()['Hash']
    hash_decode[init_state_hash] = env
    state_q.put(init_state_hash)

    # Keep track of the depth of the search (how many steps from the initial state)
    nodes_left_current_depth = 1
    nodes_in_next_level = 0
    steps_done = 0

    # Do the breadth-first search
    while not state_q.empty():
        current_state = state_q.get()
        nodes_left_current_depth -= 1

        # Being in the mapping means that the state has been visited
        if current_state not in mapping:
            mapping[current_state] = [{} for _ in action_space]
            # Add to mapping the possible follow-up states with their probabilities, per action
            for action_idx, action in enumerate(action_space):
                # If the environment is not deterministic, determine the probabilities by sampling
                for _ in range(samples):
                    # Do the stepping in a cloned environment
                    clone_env = hash_decode[current_state].clone()
                    clone_env.step(build_action(action, (current_player + steps_done) % env.player_count + 1, env.player_count))
                    next_state_hash = clone_env.get_state()['Hash']
                    
                    # Increase the probability of reaching this follow-up state from the current state by 1/samples
                    if next_state_hash not in mapping[current_state][action_idx]:
                        mapping[current_state][action_idx][next_state_hash] = 1.0 / samples
                    else:
                        mapping[current_state][action_idx][next_state_hash] += 1.0 / samples
                    if next_state_hash not in hash_decode:
                        hash_decode[next_state_hash] = clone_env

                    # Add each possible follow-up state to the queue, if they have not been visited yet
                    if next_state_hash not in mapping:
                        state_q.put(next_state_hash)
                        nodes_in_next_level += 1

        # Detect if all states have been reached in this depth (steps from the original state). If so, move to the next depth
        if nodes_left_current_depth == 0:
            steps_done += 1
            # Don't search further than the given number of steps from the original state
            if steps_done >= steps:
                return mapping
            nodes_left_current_depth = nodes_in_next_level
            nodes_in_next_level = 0    
    return mapping


def build_distribution(env_hash, mapping, action_seq, action_stepper, action_space, active_agent, current_step_agent, player_count, steps):
    '''
    Builds the distribution p(S_t+n|s_t, a_t^n)
    s_t is given by env_hash
    a_t^n is given by action_seq
    
    Params:
        env_hash defines s_t                -> the original state
        mapping defines p(S_t+1|s_t, a_t)   -> the mapping from all possible states to the next states, depending on selected action
        action_seq defines a_t^n            -> the action sequence for THE PLAYER WE ARE INTERESTED IN
        action_stepper                      -> which action is next in turn in the action sequence
        action_space                        -> possible actions of the game
        active_player                       -> player that we are interested in (for whom the action sequence applies to)
        current_step_player                 -> the player whose turn it is next, if this is the active player, then we use the action sequence. Otherwise, we build the distribution for all possible actions
        player_count                        -> How many players there are in the game
        steps                               -> How many steps forward the probability distribution should be built

    Returns:
        A dictionary, where the keys are the states and the values are the probabilities: {state_hash: probability}
    '''
    # If this is the last step, just return this state and probability 1.0
    if steps == 0:
        return {env_hash: 1.0}
    
    # If this step is for the agent whose empowerment is being calculated, take the next action from the actions list. Otherwise, we use all possible actions 0 ... len(action_space-1)
    curr_agent_actions = [action_seq[action_stepper]] if active_agent == current_step_agent else range(len(action_space))
    # Also, increase the action stepper if this agent was the active one.
    next_action_step = action_stepper + 1 if active_agent == current_step_agent else action_stepper

    state_distribution_nstep = {}
    
    # If we step forward with multiple actions, we assume uniform distribution of those actions.
    coeff = 1 / len(curr_agent_actions)
    for action in curr_agent_actions:
        # From the pre-built mapping, get the probability distribution for the next step, given an action
        next_step_probs = mapping[env_hash][action]
        # Recursively, build the distribution for each possbile follow-up state
        for next_hash, next_hash_prob in next_step_probs.items():
            next_distribution = build_distribution(next_hash, mapping, action_seq, next_action_step, action_space, active_agent, current_step_agent % player_count + 1, player_count, steps-1)
            # Add the follow-up states to the overall distribution of this state p(S_t+n|s_t, a_t^n)
            for key in next_distribution:
                if key not in state_distribution_nstep:
                    state_distribution_nstep[key] = 0
                state_distribution_nstep[key] += next_distribution[key] * coeff * next_hash_prob
    return state_distribution_nstep


def calculate_state_empowerment(state, hash_decode, mapping, action_combinations, actuator, perceptor, action_space, player_count, action_seq_length):
    # A list of end state probabilities, for each action combination. A list of dictionaries.
    states_for_action_seqs = [None] * len(action_combinations)
    for combo_idx, action_combo in enumerate(action_combinations):
        states_for_action_seqs[combo_idx] = build_distribution(state, mapping, action_combo, 0, action_space, actuator, actuator, player_count, action_seq_length * player_count)
    
    # Covert the end states into observations of the active player (actuator of the empowerment pair)
    states_as_obs = [{} for _ in states_for_action_seqs]
    for sequence_id, cpd_states in enumerate(states_for_action_seqs):
        for state_hash in cpd_states:
            latest_obs = hash_decode[state_hash]._player_last_observation[perceptor-1]
            hashed_obs = hash_obs(latest_obs)
            # Multiple states may lead to same observation, combine them if so
            if hashed_obs not in states_as_obs[sequence_id]:
                states_as_obs[sequence_id][hashed_obs] = 0
            states_as_obs[sequence_id][hashed_obs] += cpd_states[state_hash]
    # Use the Blahut-Arimoto algorithm to calculate the optimal probability distribution
    pd_a_opt = empowerment_maximization.blahut_arimoto(states_as_obs, np.random.default_rng())
    # Use the optimal distribution to calculate the mutual information (empowerement)
    empowerment = empowerment_maximization.mutual_information(pd_a_opt, states_as_obs)
    return empowerment


def calculate_expected_empowerments(env, mapping, action_seq_length, anticipation_step_count, action_space, active_agent, emp_pair, hash_decode):
    '''
    Calculates the expected empowerment for each action that can be taken from the current state, for one empowerment pair (actuator, perceptor).
    '''
    # Find the probability of each follow-up state after anticipation_step_count steps, for each action from the current state. p(S_{t+m}|s_t, a_t), m=anticipation_step_count.
    anticipation = [None] * len(action_space)
    for action_idx, action in enumerate(action_space):
        anticipation[action_idx] = build_distribution(env.get_state()['Hash'], mapping, [action_idx], 0, action_space, active_agent, active_agent, env.player_count, anticipation_step_count)
    # Make a flat list of all the reachable states
    reachable_states = []
    for a in anticipation:
        for s_hash in a:
            reachable_states.append(s_hash)
    # All possible action combinations of length 'step'
    action_combinations = [list(combo) for combo in product(range(len(action_space)), repeat=action_seq_length)]
    # Save the expected empowerment for each anticipated state
    state_empowerments = {}
    # Calculate the n-step empowerment for each state that was found earlier
    for state in reachable_states:
        empowerment = calculate_state_empowerment(state, hash_decode, mapping, action_combinations, emp_pair[0], emp_pair[1], action_space, env.player_count, action_seq_length)
        state_empowerments[state] = empowerment
    # Calculate the expected empowerment for each action that can be taken from the current state
    expected_empowerments = np.zeros(len(action_space))
    for a in range(0, len(action_space)):
        for s, p_s in anticipation[a].items():
            expected_empowerments[a] += p_s * state_empowerments[s]
    return expected_empowerments


def cem_action(env, current_player, action_seq_length, empowerment_pairs, empowerment_weights, samples=1):
    '''
    env: the game environment
    current_player: the id of the player to take an action for
    steps: the number of steps (n-step empowerment)
    empowerment_pairs: a list of tuples, where the first element is the playerId whos actuator 
        is the input and the second element is the playerId whose perceptor is the output of the communication channel
    empowerment_weights: an array of floats, where the ith element is the weight of the ith empowerment pair to be used 
        in the action policy
    samples: How many times to sample the actions to build the conditional probability distribution p(s_t+1|s_t, a). 1 for deterministic games.
    '''
    # List all possible actions in the game
    action_space = [(0, 0)] # Include the idling action
    for action_type_index, action_name in enumerate(env.action_names):
        for action_id in range(1, env.num_action_ids[action_name]):
            action_space.append((action_type_index, action_id))

    # Stores the expected empowerment for each empowerment_pair, for each action that can be taken from the current state.
    # For example: [E[E^P]_{a_t}, E[E^T]_{a_t}, E[E^C]_{a_t}], if we were to calculate three different empowerments E^P, E^T and E^C.
    # Here's an example of the structure, if there are 3 empowerment pairs (pair 0, pair 1, pair 2) and 3 actions (a0, a1, a2)
    #                   [ [1.0, 1.5, 2.0], [1.5, 1.2, 1.9], [3.0, 2.5, 1.0] ]  <- example data
    #                       |    |    |      |    |    |      |    |    |
    #                       a0   a1   a2     a0   a1   a2     a0   a1   a2    
    #                     |---pair 0----|  |----pair 1---|  |---pair 2----|
    expected_empowerments_per_pair = []

    # Saves into an array how many anticipation steps there are for each empowerment pair, i.e. how many steps before calculating the actual empowerment
    anticipation_step_counts = [(emp_pair[0] - current_player if emp_pair[0] > current_player else env.player_count - (current_player - emp_pair[0])) for emp_pair in empowerment_pairs]
    # Will contain the mapping from state hashes to states
    hash_decode = {}
    # p(S{t+1}|S_t, A_t) for all reachable states.
    mapping = build_mapping(env, hash_decode, max(anticipation_step_counts) + action_seq_length * env.player_count, action_space, current_player, samples)

    for pair_i, emp_pair in enumerate(empowerment_pairs):
        expected_empowerments = calculate_expected_empowerments(env, mapping, action_seq_length, anticipation_step_counts[pair_i], action_space, current_player, emp_pair, hash_decode)
        expected_empowerments_per_pair.append(expected_empowerments)

    EPSILON = 1e-5
    best_actions = []
    best_rewards = []
    # Find the action index that yields th highest expected empowerment
    for a in range(len(action_space)):
        policy_reward = 0
        for e in range(len(empowerment_pairs)):
            policy_reward += empowerment_weights[e] * expected_empowerments_per_pair[e][a]
        if not len(best_rewards) or policy_reward >= max(best_rewards) + EPSILON:
            best_rewards = [policy_reward]
            best_actions = [a]
        elif policy_reward > max(best_rewards) - EPSILON and policy_reward < max(best_rewards) + EPSILON:
            best_rewards.append(policy_reward)
            best_actions.append(a)
        
    return action_space[random.choice(best_actions)]
