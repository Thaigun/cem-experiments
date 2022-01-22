from itertools import product
from tracemalloc import start
import empowerment_maximization
import numpy as np
from functools import lru_cache
import random


# Hashes the numpy array of observations
def hash_obs(obs, player_pos):
    byte_repr = obs.tobytes() + bytes(player_pos)
    return hash(byte_repr)


def find_player_pos(wrapped_env, player_id):
    return list(wrapped_env.get_available_actions(player_id))[0]


def find_player_pos_vanilla(env, player_id):
    return list(env.game.get_available_actions(player_id))[0]


def find_player_health(env_state, player_id):
    # TODO: implement
    vars = next(o['Variables'] for o in env_state['Objects'] if o['Name'] == 'plr' and o['PlayerId'] == player_id)
    return vars['health']


def find_alive_players(env):
    return [player_id for player_id in range(1, env.player_count + 1) if env.get_available_actions(player_id)]


# Returns the index of the winning team (as it stands in self.teams) or -1 if no winner.
# Winning team is where one of the members satisfy Griddly's winning condition
# or if every other team is eliminated.
def find_winner(env, info, teams):
    if 'PlayerResults' in info:
        for plr, status in info['PlayerResults'].items():
            if status == 'Win':
                return next(team_i for team_i, team in enumerate(teams) if int(plr) in team)
    teams_alive = []
    alive_players = find_alive_players(env)
    for team_i, team in enumerate(teams):
        for player_id in team:
            if player_id in alive_players:
                teams_alive.append(team_i)
                if len(teams_alive) > 1:
                    return -1
                break
    if len(teams_alive) == 1:
        return teams_alive[0]
    return -1


class EnvHashWrapper():
    def __init__(self, env):
        self.env = env
        self.hash = None

    def __hash__(self) -> int:
        if self.hash is None:
            self.hash = self.env.get_state()['Hash']
        return self.hash

    def __eq__(self, __o: object) -> bool:
        return self.__hash__() == __o.__hash__()

    def get_state(self):
        return self.env.get_state()

    def get_hash(self):
        if self.hash is None:
            self.hash = self.env.get_state()['Hash']

    def clone(self):
        return EnvHashWrapper(self.env.clone())

    # We change the API here a bit, because we need this object to be immutable
    def step(self, actions):
        clone_env = self.env.clone()
        obs, rew, env_done, info = clone_env.step(actions)
        return EnvHashWrapper(clone_env), obs, rew, env_done, info

    def get_available_actions(self, player_id):
        return self.env.game.get_available_actions(player_id)

    def player_last_obs(self, player_id):
        # TODO: if this is not found, take an empty step to create the observations
        return self.env._player_last_observation[player_id - 1]

    @property
    def player_count(self):
        return self.env.player_count


class GameEndState():
    def __init__(self, winner_team):
        self.winner = winner_team

    def __hash__(self) -> int:
        return self.winner


class CEMEnv():
    def __init__(self, env, current_player, empowerment_pairs, empowerment_weights, teams, n_step, agent_actions=None, max_health=False, seed=None, samples=1):
        self.empowerment_pairs = empowerment_pairs
        self.empowerment_weights = empowerment_weights
        self.samples = samples
        self.n_step = n_step
        self.current_player = current_player
        self.env = EnvHashWrapper(env)
        self.teams = teams
        self.max_health = max_health
        self.player_count = self.env.player_count

        # List all possible actions in the game
        self.action_spaces = [[] for _ in range(self.player_count)] 
        # Include the idling action
        for player_i in range(self.player_count):
            if agent_actions is None or 'idle' in agent_actions[player_i]:
                self.action_spaces[player_i].append((0,0))
            for action_type_index, action_name in enumerate(env.action_names):
                if agent_actions is None or action_name in agent_actions[player_i]:
                    for action_id in range(1, env.num_action_ids[action_name]):
                        self.action_spaces[player_i].append((action_type_index, action_id))
        # Will contain the mapping from state hashes to states
        self.rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)


    def apply_new_state(self, new_env, current_player):
        '''
        When the game has progressed, this method can be called to update the member variables.
        This can be faster than building the new object from scratch, because for example the mapping can be reused to some extent.
        '''
        self.env = EnvHashWrapper(new_env)
        self.current_player = current_player


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


    @lru_cache(maxsize=100)
    def get_state_mapping(self, env, player_id):
        # Build it lazily
        next_agent_id = player_id % self.player_count + 1
        health_ratio = find_player_health(env.get_state(), player_id) / self.max_health if self.max_health else 1
        current_env_and_player = (env, player_id)
        result = [{} for _ in self.action_spaces[player_id-1]]
        for action_idx, action in enumerate(self.action_spaces[player_id-1]):
            for _ in range(self.samples):
                clone_env, obs, rew, env_done, info = env.step(self.build_action(action, player_id))
                game_winner = find_winner(clone_env, info, self.teams)
                if game_winner != -1:
                    next_env = GameEndState(game_winner)
                else:
                    next_env = clone_env
                next_env_and_agent = (next_env, next_agent_id)
                if next_env_and_agent not in result[action_idx]:
                    result[action_idx][next_env_and_agent] = 0
                result[action_idx][next_env_and_agent] += 1.0 / self.samples

            # Adjust fot health-performance consistency
            if self.max_health and health_ratio < 1 - 1e-5:
                for following_hash_and_agent in result[action_idx]:
                    if following_hash_and_agent[0] != current_env_and_player[0]:
                        result[action_idx][following_hash_and_agent] *= health_ratio
                no_step_hash = (current_env_and_player[0], next_agent_id)
                if no_step_hash not in result[action_idx]:
                    result[action_idx][no_step_hash] = 0
                result[action_idx][no_step_hash] = 1 - health_ratio + health_ratio * result[action_idx][no_step_hash]
        return result


    def build_distribution(self, wrapped_env, action_seq, action_stepper, active_agent, current_step_agent, perceptor):
        '''
        Builds the distribution p(S_t+n|s_t, a_t^n)
        s_t is given by wrapped_env
        a_t^n is given by action_seq
        
        Params:
            wrapped_env defines s_t             -> the original state
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
            return {wrapped_env: 1.0}

        # If this is one of the terminated states, return a mapping where each following active agent action leads to a different outcome
        if isinstance(wrapped_env, GameEndState):
            return {wrapped_env: 1.0}
        
        # If this step is for the agent whose empowerment is being calculated, take the next action from the actions list. Otherwise, we use all possible actions 0 ... len(action_space-1)
        curr_available_actions = [action_seq[action_stepper]] if active_agent == current_step_agent else range(len(self.action_spaces[current_step_agent-1]))
        # Also, increase the action stepper if this agent was the active one.
        next_action_step = action_stepper + 1 if active_agent == current_step_agent else action_stepper

        pd_s_nstep = {}
        
        # If we step forward with multiple actions, we assume uniform distribution of those actions.
        assumed_policy = 1 / len(curr_available_actions)
        for action in curr_available_actions:
            # From the pre-built mapping, get the probability distribution for the next step, given an action
            next_step_pd_s = self.get_state_mapping(wrapped_env, current_step_agent)[action]
            # Recursively, build the distribution for each possbile follow-up state
            for next_state, next_state_prob in next_step_pd_s.items():
                child_distribution = self.build_distribution(next_state[0], action_seq, next_action_step, active_agent, next_state[1], perceptor)
                # Add the follow-up states to the overall distribution of this state p(S_t+n|s_t, a_t^n)
                for next_env in child_distribution:
                    if next_env not in pd_s_nstep:
                        pd_s_nstep[next_env] = 0
                    pd_s_nstep[next_env] += child_distribution[next_env] * assumed_policy * next_state_prob
        return pd_s_nstep


    @lru_cache(maxsize=8000)
    def calculate_state_empowerment(self, env, actuator, perceptor):
        # All possible action combinations of length 'step'
        action_sequences = [list(combo) for combo in product(range(len(self.action_spaces[actuator-1])), repeat=self.n_step)]
        # A list of end state probabilities, for each action combination. A list of dictionaries.
        cpd_S_A_nstep = [None] * len(action_sequences)
        for seq_idx, action_seq in enumerate(action_sequences):
            cpd_S_A_nstep[seq_idx] = self.build_distribution(env, action_seq, 0, actuator, actuator, perceptor)
        
        # Covert the end states into observations of the active player (actuator of the empowerment pair)
        states_as_obs = [{} for _ in cpd_S_A_nstep]
        for sequence_id, pd_states in enumerate(cpd_S_A_nstep):
            for wrapped_env in pd_states:
                if isinstance(wrapped_env, GameEndState):
                    if actuator in self.teams[wrapped_env.winner]:
                        hashed_obs = self.rng.integers(self.player_count + 1, 4000000000)
                    else:
                        hashed_obs = wrapped_env.winner
                else:
                    latest_obs = wrapped_env.player_last_obs(perceptor)
                    player_pos = find_player_pos(wrapped_env, perceptor)
                    hashed_obs = hash_obs(latest_obs, player_pos)
                # Multiple states may lead to same observation, combine them if so
                if hashed_obs not in states_as_obs[sequence_id]:
                    states_as_obs[sequence_id][hashed_obs] = 0
                states_as_obs[sequence_id][hashed_obs] += pd_states[wrapped_env]
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
            cpd_s_a_anticipation[action_idx] = self.build_distribution(self.env, [action_idx], 0, self.current_player, self.current_player, emp_pair[0])
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
        print(self.get_state_mapping.cache_info())
        return expected_empowerments
