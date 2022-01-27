from itertools import product
import empowerment_maximization
import numpy as np
from functools import lru_cache
import random
from collections import namedtuple


EmpConf = namedtuple('EmpConf', ['empowerment_pairs', 'empowerment_weights'])


# Hashes the numpy array of observations
def hash_obs(obs, player_pos):
    if player_pos is None:
        return hash(b'a')
    return hash(obs.tobytes() + bytes(player_pos))


def find_player_pos(wrapped_env, player_id):
    agent_actions = wrapped_env.get_available_actions(player_id)
    if not agent_actions:
        return None
    return list(wrapped_env.get_available_actions(player_id))[0]


def find_player_pos_vanilla(env, player_id):
    return list(env.game.get_available_actions(player_id))[0]


def find_player_health(env_state, player_id):
    player_variables = [o['Variables'] for o in env_state['Objects'] if o['Name'] == 'plr' and o['PlayerId'] == player_id]
    if not player_variables:
        return 0
    return player_variables[0]['health']


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
    # This is a bit dangerous, but the env that is given to the constructor must not be stepped.
    # If we can find a way to not lose too much performance by cloning it here in the constructor, that would be perfect.
    def __init__(self, env):
        self._env = env
        self.hash = None

    def __hash__(self) -> int:
        return self.get_hash()

    def __eq__(self, __o: object) -> bool:
        return self.__hash__() == __o.__hash__()

    def get_state(self):
        return self._env.get_state()

    def get_hash(self):
        if self.hash is None:
            self.hash = self._env.get_state()['Hash']
        return self.hash

    def clone(self):
        return EnvHashWrapper(self._env.clone())

    # We change the API here a bit, because we need this object to be immutable
    def step(self, actions):
        clone_env = self._env.clone()
        obs, rew, env_done, info = clone_env.step(actions)
        return EnvHashWrapper(clone_env), obs, rew, env_done, info

    def get_available_actions(self, player_id):
        return self._env.game.get_available_actions(player_id)

    def player_last_obs(self, player_id):
        if not self._env._player_last_observation:
            self._env.step([[0, 0] for _ in range(self.player_count)])
        return self._env._player_last_observation[player_id - 1]

    @property
    def player_count(self):
        return self._env.player_count


class GameEndState():
    def __init__(self, winner_team):
        self.winner = winner_team

    def __hash__(self) -> int:
        return self.winner

    def __eq__(self, __o: object) -> bool:
        try:
            return self.winner == __o.winner
        except:
            return False


class CEM():
    def __init__(self, env, empowerment_confs, teams, agent_actions=None, max_healths=False, seed=None, samples=1):
        self.empowerment_confs = empowerment_confs
        self.samples = samples
        self.teams = teams
        self.max_healths = max_healths
        self.player_count = env.player_count

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


    def cem_action(self, env, player_id, n_step):
        # Stores the expected empowerment for each empowerment_pair, for each action that can be taken from the current state.
        # For example: [E[E^P]_{a_t}, E[E^T]_{a_t}, E[E^C]_{a_t}], if we were to calculate three different empowerments E^P, E^T and E^C.
        # Here's an example of the structure, if there are 3 empowerment pairs (pair 0, pair 1, pair 2) and 3 actions (a0, a1, a2)
        #                   [ [1.0, 1.5, 2.0], [1.5, 1.2, 1.9], [3.0, 2.5, 1.0] ]  <- example data
        #                       |    |    |      |    |    |      |    |    |
        #                       a0   a1   a2     a0   a1   a2     a0   a1   a2    
        #                     |---pair 0----|  |----pair 1---|  |---pair 2----|
        expected_empowerments_per_pair = []

        for emp_pair in self.empowerment_confs[player_id].empowerment_pairs:
            expected_empowerments = self.calculate_expected_empowerments(env, player_id, emp_pair, n_step)
            expected_empowerments_per_pair.append(expected_empowerments)

        EPSILON = 1e-5
        best_actions = []
        best_rewards = []
        # Find the action index that yields the highest expected empowerment
        for a in range(len(self.action_spaces[player_id-1])):
            policy_reward = 0
            for e in range(len(self.empowerment_confs[player_id].empowerment_pairs)):
                policy_reward += self.empowerment_confs[player_id].empowerment_weights[e] * expected_empowerments_per_pair[e][a]
            if not len(best_rewards) or policy_reward >= max(best_rewards) + EPSILON:
                best_rewards = [policy_reward]
                best_actions = [a]
            elif policy_reward > max(best_rewards) - EPSILON and policy_reward < max(best_rewards) + EPSILON:
                best_rewards.append(policy_reward)
                best_actions.append(a)
        
        return self.action_spaces[player_id-1][random.choice(best_actions)]


    # Builds the action that can be passed to the Griddly environment
    def build_action(self, action, player_id):
        built_action = [[0,0] for _ in range(self.player_count)]
        built_action[player_id-1] = list(action)
        return built_action


    @lru_cache(maxsize=5000)
    def calc_cpd_s_a(self, env, player_id, action):
        # Build it lazily
        health_ratio = find_player_health(env.get_state(), player_id) / self.max_healths[player_id-1] if self.max_healths else 1
        # if the player is dead, the empowerment is 0
        if health_ratio == 0:
            return {env: 1.0}

        result = {}
        for _ in range(self.samples):
            clone_env, obs, rew, env_done, info = env.step(self.build_action(action, player_id))
            if (env_done):
                game_winner = find_winner(clone_env, info, self.teams)
                if game_winner != -1:
                    next_env = GameEndState(game_winner)
            else:
                next_env = clone_env
            if next_env not in result:
                result[next_env] = 0
            result[next_env] += 1.0 / self.samples

        # Adjust for health-performance consistency
        if self.max_healths and health_ratio < 1 - 1e-5:
            self.health_perf_consistency(env, health_ratio, result)
        return result


    def health_perf_consistency(self, env, health_ratio, transitions):
        # If the following env is not the current_env, adjust the transition probability with health_ratio
        for next_env in transitions:
            if next_env != env:
                transitions[next_env] *= health_ratio
        # Add the current env to the possible transitions
        if env not in transitions:
            transitions[env] = 0
        # Probability of staying still is its original probability and "leftovers" of the other states.
        transitions[env] = 1 - health_ratio + health_ratio * transitions[env]


    #@lru_cache(maxsize=2000)
    def build_distribution(self, wrapped_env, action_seq, action_stepper, active_agent, current_step_agent, perceptor, return_obs=False, trust_correction_steps=None):
        '''
        Recursive function
        Builds the distribution p(S_t+n|s_t, a_t^n)
        s_t is given by wrapped_env
        a_t^n is given by action_seq
        
        Params:
            wrapped_env defines s_t             -> the original state, wrapped in a hasher
            action_seq defines a_t^n            -> the action sequence for THE PLAYER WE ARE INTERESTED IN
            action_stepper                      -> which action is next in turn in the action sequence
            active_agent                        -> player that we are interested in (for whom the action sequence applies to)
            current_step_player                 -> the player whose turn it is next, if this is the active player, then we use the action sequence. Otherwise, we build the distribution for all possible actions
            perceptor                           -> the agent until whose perception the distribution is built, after all actions have been taken
            return_obs                          -> if true, we don't return the end states but corresponding observations
            trust_correction_steps              -> an array of boolean values, indicating which of the following steps should be trust_corrected

        Returns:
            A dictionary, where the keys are the states(/observations) and the values are the probabilities: {state: probability}
        '''
        # If this is the last step, just return this state and probability 1.0
        # or if this is one of the terminated states, return a mapping where each following active agent action leads to a different outcome
        if (action_stepper == len(action_seq) and current_step_agent == perceptor) or isinstance(wrapped_env, GameEndState):
            # if return_obs is True, then we return the observation instead of the state
            return {self.env_to_hashed_obs(wrapped_env, active_agent, perceptor): 1.0} if return_obs else {wrapped_env: 1.0}
        
        # If this step is for the agent whose empowerment is being calculated, take the next action from the actions list. Otherwise, we use all possible actions 0 ... len(action_space-1)
        curr_available_actions = [action_seq[action_stepper]] if active_agent == current_step_agent else range(len(self.action_spaces[current_step_agent-1]))
        # Also, increase the action stepper if this agent was the active one.
        next_action_step = action_stepper + 1 if active_agent == current_step_agent else action_stepper

        pd_s_nstep = {}
        active_agent_team = next(team for team in self.teams if active_agent in team)

        # If we step forward with multiple actions, we assume uniform distribution of those actions.
        assumed_policy = 1.0 / len(curr_available_actions)
        # Sum up the total probability of all possible actions so we can normalize in the end if needed
        total_prob = 0
        for action in curr_available_actions:
            # Get the probability distribution for the next step, given an action
            next_step_pd_s = self.calc_cpd_s_a(wrapped_env, current_step_agent, self.action_spaces[current_step_agent-1][action])
            correct_for_trust = trust_correction_steps and len(trust_correction_steps) > 0 and trust_correction_steps[0]

            # Recursively, build the distribution for each possbile follow-up state
            for next_state, next_state_prob in next_step_pd_s.items():
                next_step_agent = current_step_agent % self.player_count + 1
                following_trust_correction = trust_correction_steps[1:] if trust_correction_steps else trust_correction_steps
                # If the follow-up state has zero empowerment for the active agent and the current state doesn't we skip the action
                # Because we assume the player in the same team wouldn't reduce our empowerment to zero.
                if correct_for_trust and current_step_agent in active_agent_team and next_state_prob > 0.01:
                    follow_up_emp = self.calculate_state_empowerment(next_state, active_agent, active_agent, 1)
                    if follow_up_emp == 0:
                        current_state_emp = self.calculate_state_empowerment(wrapped_env, active_agent, active_agent, 1)
                        if current_state_emp != 0:
                            break
            
                child_distribution = self.build_distribution(next_state, action_seq, next_action_step, active_agent, next_step_agent, perceptor, return_obs, following_trust_correction)
                # Add the follow-up states to the overall distribution of this state p(S_t+n|s_t, a_t^n)
                for next_env in child_distribution:
                    if next_env not in pd_s_nstep:
                        pd_s_nstep[next_env] = 0
                    adjusted_prob = child_distribution[next_env] * assumed_policy * next_state_prob
                    pd_s_nstep[next_env] += adjusted_prob
                    total_prob += adjusted_prob
        if total_prob > 0 and total_prob < 1 - 1e-5:
            for state in pd_s_nstep:
                pd_s_nstep[state] /= total_prob
        return pd_s_nstep


    @lru_cache(maxsize=8000)
    def calculate_state_empowerment(self, wrapped_env, actuator, perceptor, n_step, trust_correction_steps=None):
        # If the player isn't alive anymore, we assume the empowerment to be zero
        if find_player_pos(wrapped_env, actuator) is None:
            return 0

        # All possible action combinations of length 'step'
        action_sequences = [tuple(combo) for combo in product(range(len(self.action_spaces[actuator-1])), repeat=n_step)]
        # A list of end state probabilities, for each action combination. A list of dictionaries.
        cpd_S_A_nstep = [None] * len(action_sequences)
        for seq_idx, action_seq in enumerate(action_sequences):
            cpd_S_A_nstep[seq_idx] = self.build_distribution(wrapped_env, action_seq, 0, actuator, actuator, perceptor, True, trust_correction_steps)
            
        # Use the Blahut-Arimoto algorithm to calculate the optimal probability distribution
        pd_a_opt = empowerment_maximization.blahut_arimoto(cpd_S_A_nstep, self.rng)
        # Use the optimal distribution to calculate the mutual information (empowerement)
        empowerment = empowerment_maximization.mutual_information(pd_a_opt, cpd_S_A_nstep)
        return empowerment


    def calculate_expected_empowerments(self, gym_env, current_player, emp_pair, n_step, trust_correction_steps=None):
        '''
        Calculates the expected empowerment for each action that can be taken from the current state, for one empowerment pair (actuator, perceptor).
        '''
        # Find the probability of follow-up states for when the actuator is in turn. p(s|s_t, a_t), s = state when actuator is in turn
        cpd_s_a_anticipation = [None] * len(self.action_spaces[current_player-1])
        for action_idx, action in enumerate(self.action_spaces[current_player-1]):
            cpd_s_a_anticipation[action_idx] = self.build_distribution(EnvHashWrapper(gym_env.clone()), (action_idx,), 0, current_player, current_player, emp_pair[0], trust_correction_steps=trust_correction_steps)
        # Make a flat list of all the reachable states
        reachable_states = set()
        for cpd_s in cpd_s_a_anticipation:
            for future_state in cpd_s:
                reachable_states.add(future_state)
        # Save the expected empowerment for each anticipated state
        reachable_state_empowerments = {}
        # Calculate the n-step empowerment for each state that was found earlier
        for state in reachable_states:
            empowerment = self.calculate_state_empowerment(state, emp_pair[0], emp_pair[1], n_step)
            reachable_state_empowerments[state] = empowerment
        # Calculate the expected empowerment for each action that can be taken from the current state
        expected_empowerments = np.zeros(len(self.action_spaces[current_player-1]))
        for a in range(0, len(self.action_spaces[current_player-1])):
            for state, state_probability in cpd_s_a_anticipation[a].items():
                expected_empowerments[a] += state_probability * reachable_state_empowerments[state]
        
        return expected_empowerments

    
    def env_to_hashed_obs(self, wrapped_env, actuator, perceptor):
        if isinstance(wrapped_env, GameEndState):
            # If the actuator is in the winning team, all their actions lead to maximum empowerment
            if actuator in self.teams[wrapped_env.winner]:
                hashed_obs = self.rng.integers(100, 4000000000)
            # But if the actuator loses futura actions lead to a minimum empowerment
            else:
                hashed_obs = wrapped_env.winner
        else:
            latest_obs = wrapped_env.player_last_obs(perceptor)
            player_pos = find_player_pos(wrapped_env, perceptor)
            # If the player was not found, we assume the player is dead. In that case, the hashed observation is the player's id
            hashed_obs = hash_obs(latest_obs, player_pos) if player_pos is not None else hash(actuator)
        return hashed_obs
