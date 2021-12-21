from itertools import product
import empowerment_maximization
import numpy as np
import queue
import random


# Hashes the numpy array of observations
def hash_obs(obs, player_pos):
    byte_repr = obs.tobytes() + bytes(player_pos)
    return hash(byte_repr)


def find_player_pos(env, player_id):
    return list(env.game.get_available_actions(player_id))[0]


def find_player_health(env_state, player_id):
    # TODO: implement
    vars = next(o['Variables'] for o in env_state['Objects'] if o['Name'] == 'plr' and o['PlayerId'] == player_id)
    return vars['health']


class CEMEnv():
    def __init__(self, env, current_player, empowerment_pairs, empowerment_weights, teams, n_step, agent_actions=None, max_health=False, seed=None, samples=1):
        self.empowerment_pairs = empowerment_pairs
        self.empowerment_weights = empowerment_weights
        self.samples = samples
        self.n_step = n_step
        self.current_player = current_player
        self.env = env
        self.teams = teams
        self.max_health = max_health

        # List all possible actions in the game
        self.action_spaces = [[] for _ in range(env.player_count)] 
        # Include the idling action
        for player_i in range(env.player_count):
            if agent_actions is None or 'idle' in agent_actions[player_i]:
                self.action_spaces[player_i].append((0,0))
            for action_type_index, action_name in enumerate(env.action_names):
                if agent_actions is None or action_name in agent_actions[player_i]:
                    for action_id in range(1, env.num_action_ids[action_name]):
                        self.action_spaces[player_i].append((action_type_index, action_id))
        # Will contain the mapping from state hashes to states
        self.hash_decode = {}
        self.rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        self.player_count = env.player_count
        # Sn array of how many anticipation steps there are for each empowerment pair, i.e. how many steps before calculating the actual empowerment
        anticipation_step_counts = [self.calc_anticipation_step_count(current_player, empowerment_pair[0]) for empowerment_pair in empowerment_pairs]
        # p(S{t+1}|S_t, A_t) for all reachable states.
        self.mapping = self.build_mapping(max(anticipation_step_counts) + n_step * self.player_count, samples)


    def apply_new_state(self, new_state):
        '''
        When the game has progressed, this method can be called to update the member variables.
        This can be faster than building the new object from scratch, because for example the mapping is not entirely recalculated.
        TODO: Implement!
        '''
        pass


    def calc_anticipation_step_count(self, curr_plr, actuator):
        return (actuator - curr_plr) if actuator > curr_plr else self.player_count - (curr_plr - actuator)


    def cem_action(self):
        # Stores the expected empowerment for each empowerment_pair, for each action that can be taken from the current state.
        # For example: [E[E^P]_{a_t}, E[E^T]_{a_t}, E[E^C]_{a_t}], if we were to calculate three different empowerments E^P, E^T and E^C.
        # Here's an example of the structure, if there are 3 empowerment pairs (pair 0, pair 1, pair 2) and 3 actions (a0, a1, a2)
        #                   [ [1.0, 1.5, 2.0], [1.5, 1.2, 1.9], [3.0, 2.5, 1.0] ]  <- example data
        #                       |    |    |      |    |    |      |    |    |
        #                       a0   a1   a2     a0   a1   a2     a0   a1   a2    
        #                     |---pair 0----|  |----pair 1---|  |---pair 2----|
        expected_empowerments_per_pair = []

        for emp_pair in self.empowerment_pairs:
            expected_empowerments = self.calculate_expected_empowerments(emp_pair)
            expected_empowerments_per_pair.append(expected_empowerments)

        EPSILON = 1e-5
        best_actions = []
        best_rewards = []
        # Find the action index that yields the highest expected empowerment
        for a in range(len(self.action_spaces[self.current_player-1])):
            policy_reward = 0
            for e in range(len(self.empowerment_pairs)):
                policy_reward += self.empowerment_weights[e] * expected_empowerments_per_pair[e][a]
            if not len(best_rewards) or policy_reward >= max(best_rewards) + EPSILON:
                best_rewards = [policy_reward]
                best_actions = [a]
            elif policy_reward > max(best_rewards) - EPSILON and policy_reward < max(best_rewards) + EPSILON:
                best_rewards.append(policy_reward)
                best_actions.append(a)
        
        return self.action_spaces[self.current_player-1][random.choice(best_actions)]


    # Builds the action that can be passed to the Griddly environment
    def build_action(self, action, player_id):
        built_action = [[0,0] for _ in range(self.player_count)]
        built_action[player_id-1] = list(action)
        return built_action


    # Do a breadth first search on the env to build the mapping from each state to the next states with probabilities. The mapping covers all states that are reachable in 'steps' steps.
    def build_mapping(self, steps, samples=1):
        mapping = {}
        state_q = queue.Queue()

        # Initialize the breadth-first search
        init_state_hash = self.env.get_state()['Hash']
        self.hash_decode[init_state_hash] = self.env
        state_q.put((init_state_hash, self.current_player))

        # Keep track of the depth of the search (how many steps from the initial state)
        nodes_left_current_depth = 1
        nodes_in_next_level = 0
        steps_done = 0

        # Do the breadth-first search
        while not state_q.empty():
            current_hash_and_player = state_q.get()
            nodes_left_current_depth -= 1

            # Being in the mapping means that the state has been visited
            if current_hash_and_player not in mapping:
                current_env = self.hash_decode[current_hash_and_player[0]]
                curr_agent_id = current_hash_and_player[1]
                next_agent_id = (self.current_player + steps_done) % self.player_count + 1
                health_ratio = find_player_health(current_env.get_state(), curr_agent_id) / self.max_health if self.max_health else 1
                mapping[current_hash_and_player] = [{} for _ in self.action_spaces[curr_agent_id-1]]
                # Add to mapping the possible follow-up states with their probabilities, per action
                for action_idx, action in enumerate(self.action_spaces[curr_agent_id-1]):
                    # If the environment is not deterministic, determine the probabilities by sampling
                    for _ in range(samples):
                        # Do the stepping in a cloned environment
                        clone_env = current_env.clone()
                        obs, rew, env_done, info = clone_env.step(self.build_action(action, curr_agent_id))
                        # If the game ends, the state is saved as simply the winner id.
                        if env_done:
                            for plr, status in info['PlayerResults'].items():
                                if status == 'Win':
                                    next_state_hash = int(plr)
                                    break
                        else:
                            next_state_hash = clone_env.get_state()['Hash']
                        
                        next_state_and_agent = (next_state_hash, next_agent_id)
                        # Increase the probability of reaching this follow-up state from the current state by 1/samples
                        if next_state_and_agent not in mapping[current_hash_and_player][action_idx]:
                            mapping[current_hash_and_player][action_idx][next_state_and_agent] = 0
                        mapping[current_hash_and_player][action_idx][next_state_and_agent] += 1.0 / samples
                        
                        if next_state_hash not in self.hash_decode and not env_done:
                            self.hash_decode[next_state_hash] = clone_env

                        # Add each possible follow-up state to the queue, if they have not been visited yet
                        if next_state_and_agent not in mapping:
                            if env_done:
                                mapping[next_state_and_agent] = [{next_state_and_agent: 1.0} for _ in self.action_spaces[next_agent_id-1]]
                            else:
                                state_q.put(next_state_and_agent)
                                nodes_in_next_level += 1

                    # Adjust fot health-performance consistency
                    if self.max_health and health_ratio < 1 - 1e-5:
                        for following_hash_and_agent in mapping[current_hash_and_player][action_idx]:
                            if following_hash_and_agent[0] != current_hash_and_player[0]:
                                mapping[current_hash_and_player][action_idx][following_hash_and_agent] *= health_ratio
                        no_step_hash = (current_hash_and_player[0], next_agent_id)
                        if no_step_hash not in mapping[current_hash_and_player][action_idx]:
                            mapping[current_hash_and_player][action_idx][no_step_hash] = 0
                        mapping[current_hash_and_player][action_idx][no_step_hash] = 1 - health_ratio + health_ratio * mapping[current_hash_and_player][action_idx][no_step_hash]

            # Detect if all states have been reached in this depth (steps from the original state). If so, move to the next depth
            if nodes_left_current_depth == 0:
                steps_done += 1
                # Don't search further than the given number of steps from the original state
                if steps_done >= steps:
                    return mapping
                nodes_left_current_depth = nodes_in_next_level
                nodes_in_next_level = 0    
        return mapping


    def build_distribution(self, env_hash, action_seq, action_stepper, active_agent, current_step_agent, perceptor):
        '''
        Builds the distribution p(S_t+n|s_t, a_t^n)
        s_t is given by env_hash
        a_t^n is given by action_seq
        
        Params:
            env_hash defines s_t                -> the original state
            action_seq defines a_t^n            -> the action sequence for THE PLAYER WE ARE INTERESTED IN
            action_stepper                      -> which action is next in turn in the action sequence
            active_agent                        -> player that we are interested in (for whom the action sequence applies to)
            current_step_player                 -> the player whose turn it is next, if this is the active player, then we use the action sequence. Otherwise, we build the distribution for all possible actions
            perceptor                           -> the agent until whose perception the distribution is built, after all actions have been taken

        Returns:
            A dictionary, where the keys are the states and the values are the probabilities: {state_hash: probability}
        '''
        # If this is the last step, just return this state and probability 1.0
        if action_stepper == len(action_seq) and current_step_agent == perceptor:
            return {env_hash: 1.0}

        # If this is one of the terminated states, return a mapping where each following active agent action leads to a different outcome
        if env_hash <= self.player_count and env_hash >= 0:
            active_agent_team = next(team for team in self.teams if active_agent in team)
            if env_hash in active_agent_team:
                # Random state hash with probability 1.0, represents a unique state
                return {self.rng.integers(self.player_count + 1): 1.0}
            else:
                return {0: 1.0}
        
        # If this step is for the agent whose empowerment is being calculated, take the next action from the actions list. Otherwise, we use all possible actions 0 ... len(action_space-1)
        curr_available_actions = [action_seq[action_stepper]] if active_agent == current_step_agent else range(len(self.action_spaces[current_step_agent-1]))
        # Also, increase the action stepper if this agent was the active one.
        next_action_step = action_stepper + 1 if active_agent == current_step_agent else action_stepper

        pd_s_nstep = {}
        
        # If we step forward with multiple actions, we assume uniform distribution of those actions.
        assumed_policy = 1 / len(curr_available_actions)
        for action in curr_available_actions:
            # From the pre-built mapping, get the probability distribution for the next step, given an action
            next_step_pd_s = self.mapping[(env_hash, current_step_agent)][action]
            # Recursively, build the distribution for each possbile follow-up state
            for next_state, next_state_prob in next_step_pd_s.items():
                child_distribution = self.build_distribution(next_state[0], action_seq, next_action_step, active_agent, next_state[1], perceptor)
                # Add the follow-up states to the overall distribution of this state p(S_t+n|s_t, a_t^n)
                for hash in child_distribution:
                    if hash not in pd_s_nstep:
                        pd_s_nstep[hash] = 0
                    pd_s_nstep[hash] += child_distribution[hash] * assumed_policy * next_state_prob
        return pd_s_nstep


    def calculate_state_empowerment(self, state_hash, actuator, perceptor):
        # All possible action combinations of length 'step'
        action_sequences = [list(combo) for combo in product(range(len(self.action_spaces[actuator-1])), repeat=self.n_step)]
        # A list of end state probabilities, for each action combination. A list of dictionaries.
        cpd_S_A_nstep = [None] * len(action_sequences)
        for seq_idx, action_seq in enumerate(action_sequences):
            cpd_S_A_nstep[seq_idx] = self.build_distribution(state_hash, action_seq, 0, actuator, actuator, perceptor)
        
        # Covert the end states into observations of the active player (actuator of the empowerment pair)
        states_as_obs = [{} for _ in cpd_S_A_nstep]
        for sequence_id, pd_states in enumerate(cpd_S_A_nstep):
            for state_hash in pd_states:
                if state_hash <= self.player_count and state_hash >= 0:
                    hashed_obs = state_hash
                else:
                    end_env = self.hash_decode[state_hash]
                    latest_obs = end_env._player_last_observation[perceptor-1]
                    player_pos = find_player_pos(end_env, perceptor)
                    hashed_obs = hash_obs(latest_obs, player_pos)
                # Multiple states may lead to same observation, combine them if so
                if hashed_obs not in states_as_obs[sequence_id]:
                    states_as_obs[sequence_id][hashed_obs] = 0
                states_as_obs[sequence_id][hashed_obs] += pd_states[state_hash]
        # Use the Blahut-Arimoto algorithm to calculate the optimal probability distribution
        pd_a_opt = empowerment_maximization.blahut_arimoto(states_as_obs, self.rng)
        # Use the optimal distribution to calculate the mutual information (empowerement)
        empowerment = empowerment_maximization.mutual_information(pd_a_opt, states_as_obs)
        return empowerment


    def calculate_expected_empowerments(self, emp_pair):
        '''
        Calculates the expected empowerment for each action that can be taken from the current state, for one empowerment pair (actuator, perceptor).
        '''
        # Find the probability of follow-up states for when the actuator is in turn. p(s|s_t, a_t), s = state when actuator is in turn
        cpd_s_a_anticipation = [None] * len(self.action_spaces[self.current_player-1])
        for action_idx, action in enumerate(self.action_spaces[self.current_player-1]):
            cpd_s_a_anticipation[action_idx] = self.build_distribution(self.env.get_state()['Hash'], [action_idx], 0, self.current_player, self.current_player, emp_pair[0])
        # Make a flat list of all the reachable states
        reachable_states = set()
        for cpd_s in cpd_s_a_anticipation:
            for s_hash in cpd_s:
                reachable_states.add(s_hash)
        # Save the expected empowerment for each anticipated state
        reachable_state_empowerments = {}
        # Calculate the n-step empowerment for each state that was found earlier
        for state in reachable_states:
            empowerment = self.calculate_state_empowerment(state, emp_pair[0], emp_pair[1])
            reachable_state_empowerments[state] = empowerment
        # Calculate the expected empowerment for each action that can be taken from the current state
        expected_empowerments = np.zeros(len(self.action_spaces[self.current_player-1]))
        for a in range(0, len(self.action_spaces[self.current_player-1])):
            for state, state_probability in cpd_s_a_anticipation[a].items():
                expected_empowerments[a] += state_probability * reachable_state_empowerments[state]
        return expected_empowerments
