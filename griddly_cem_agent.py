from itertools import product
import empowerment_maximization
import numpy as np
from functools import lru_cache
import random
import global_configuration
import env_util


EPSILON = 1e-5


# Hashes the numpy array of observations
def hash_obs(obs, player_pos):
    if player_pos is None:
        return hash(b'a')
    return hash(obs.tobytes() + bytes(player_pos))


@lru_cache(maxsize=2000)
def find_player_pos(wrapped_env, player_id):
    agent_actions = wrapped_env.get_available_actions(player_id)
    if not agent_actions:
        return None
    return list(wrapped_env.get_available_actions(player_id))[0]


def find_player_pos_vanilla(env, player_id):
    return list(env.game.get_available_actions(player_id))[0]


def find_player_health(env_state, player_id):
    player_variables = [o['Variables'] for o in env_state['Objects'] if o['Name'] == 'avatar' and o['PlayerId'] == player_id]
    if not player_variables:
        return 0
    # If health is not an object variable, return None
    if 'health' not in player_variables[0]:
        return None
    return player_variables[0]['health']


@lru_cache(maxsize=2000)
def find_alive_players(wrapped_env):
    return [player_id for player_id in range(1, wrapped_env.player_count + 1) if wrapped_env.get_available_actions(player_id)]


class EnvHashWrapper():
    def __init__(self, env):
        # The environment is very likely going to be stepped. Thus, we don't want to use the original environment here.
        if not env._is_clone:
            raise ValueError('The environment given to EnvHashWrapper must be a clone of an original environment.')
        self._env = env
        self._hash = None
        self.finished = False
        self.winner = 0

    def __hash__(self) -> int:
        return self.get_hash()

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, EnvHashWrapper) and self.__hash__() == __o.__hash__()

    def get_state(self):
        return self._env.get_state()

    def get_hash(self):
        if self._hash is None:
            if self.finished and self.winner > 0:
                self._hash = hash(self.winner)
            else:
                self._hash = self._env.get_state()['Hash']
        return self._hash

    def clone(self):
        return EnvHashWrapper(self._env.clone())

    def set_winner(self, winner):
        self.finished = True
        self.winner = winner
        self._hash = None

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


class CEM():
    def __init__(self, env, game_conf, seed=None, samples=1):
        self.samples = samples
        self.player_count = env.player_count
        self.game_conf = game_conf
        self.action_spaces = env_util.build_action_spaces(env, self.agent_confs)
        self.rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)


    @property
    def agent_confs(self):
        return self.game_conf.agents


    def cem_action(self, env, player_id, n_step):
        # If all empowerments are weighted to zero, all actions are equal.
        emp_pairs = self.get_emp_pairs(player_id)
        all_weights_zero = all([emp_pair.weight == 0 for emp_pair in emp_pairs])
        if all_weights_zero:
            best_actions = self.action_spaces[player_id-1]
        else:
            expected_empowerments_per_pair = self.calculate_expected_empowerments_per_pair(env, player_id, n_step)
            best_actions = self.find_best_actions(player_id, expected_empowerments_per_pair)
        return random.choice(best_actions)


    def get_emp_pairs(self, player_id):
        return self.agent_confs[player_id-1].empowerment_pairs


    def calculate_expected_empowerments_per_pair(self, env, player_id, n_step):
        '''
        Returns the expected empowerment for each empowerment_pair, for each action that can be taken from the current state.
        For example: [E[E^P]_{a_t}, E[E^T]_{a_t}, E[E^C]_{a_t}], if we were to calculate three different empowerments E^P, E^T and E^C.
        Here's an example of the structure, if there are 3 empowerment pairs (pair 0, pair 1, pair 2) and 3 actions (a0, a1, a2)
                          [ [1.0, 1.5, 2.0], [1.5, 1.2, 1.9], [3.0, 2.5, 1.0] ]  <- example data
                              |    |    |      |    |    |      |    |    |
                              a0   a1   a2     a0   a1   a2     a0   a1   a2    
                            |---pair 0----|  |----pair 1---|  |---pair 2----|
        '''
        expected_empowerments_per_pair = []
        for emp_pair in self.get_emp_pairs(player_id):
            expected_empowerments = self.calculate_expected_empowerments(env, player_id, (emp_pair.actor, emp_pair.perceptor), n_step, True)
            expected_empowerments_per_pair.append(expected_empowerments)
        return expected_empowerments_per_pair

        
    def find_best_actions(self, player_id, expected_empowerments_per_pair):
        best_actions = []
        best_reward = -1e10
        for action in self.action_spaces[player_id-1]:
            empowerment_pair_confs = enumerate(self.get_emp_pairs(player_id))
            action_reward = self.calculate_action_reward(action, empowerment_pair_confs, expected_empowerments_per_pair)
            if action_reward >= best_reward + EPSILON:
                best_actions = [action]
            elif abs(action_reward-best_reward) < EPSILON:
                best_actions.append(action)
            best_reward = max(best_reward, action_reward)
            if global_configuration.verbose_calculation:
                print(f'Action {action} has expected reward {action_reward}')
        return best_actions
            
        
    def calculate_action_reward(self, action, empowerment_pair_confs, expected_empowerments_per_pair):
        action_reward = 0
        for emp_pair_i, emp_conf in empowerment_pair_confs:
            action_reward += emp_conf.weight * expected_empowerments_per_pair[emp_pair_i][action]
        return action_reward


    @lru_cache(maxsize=1000)
    def calc_pd_s_a(self, env, player_id, action):
        health_ratio = self.get_health_ratio(env, player_id)
        # if the player is dead, the empowerment is 0
        if health_ratio == 0:
            return {env: 1.0}

        result = {}
        for _ in range(self.samples):
            next_env, obs, rew, env_done, info = env.step(env_util.build_action(action, self.player_count, player_id))
            if (env_done):
                game_winner = env_util.find_winner(info)
                next_env.set_winner(game_winner)
            if next_env not in result:
                result[next_env] = 0
            result[next_env] += 1.0 / self.samples

        # Adjust for health-performance consistency
        if health_ratio < 1 - EPSILON:
            self.health_perf_consistency(env, health_ratio, result)
        return result


    def get_health_ratio(self, env, player_id):
        if self.game_conf.health_performance_consistency:
            player_health = find_player_health(env.get_state(), player_id)
            if player_health is not None:
                health_ratio = player_health / self.agent_confs[player_id-1].max_health
            else:
                health_ratio = 1.0
        else:
            health_ratio = 1.0
        return health_ratio


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


    def find_assumed_policy(self, agent_id, wrapped_env):
        current_agent_conf = self.agent_confs[agent_id-1]
        # Find the policy of the agent in turn
        return current_agent_conf.assumed_policy(wrapped_env._env , self, agent_id, self.game_conf)


    def build_distribution(self, wrapped_env, action_seq, action_stepper, actor, current_step_agent, perceptor, return_obs=False, anticipation=False, trust_correction=False):
        '''
        Recursive function
        Builds the distribution p(S_t+n|s_t, a_t^n)
        s_t is given by wrapped_env
        a_t^n is given by action_seq
        
        Params:
            wrapped_env defines s_t             -> the original state, wrapped in a hasher
            action_seq defines a_t^n            -> the action sequence for THE PLAYER WE ARE INTERESTED IN
            action_stepper                      -> which action is next in turn in the action sequence
            actor                               -> player that we are interested in (for whom the action sequence applies to)
            current_step_player                 -> the player whose turn it is next, if this is the active player, then we use the action sequence. Otherwise, we build the distribution for all possible actions
            perceptor                           -> the agent until whose perception the distribution is built, after all actions have been taken
            return_obs                          -> if true, we don't return the end states but corresponding observations
            anticipation                        -> boolean value indicating whether this is an anticipation step or not
            trust_correction                    -> boolean value indicating whether we should correct for trust

        Returns:
            A dictionary, where the keys are the states(/observations) and the values are the probabilities: {state: probability}
        '''
        def is_final_state():
            return (action_stepper == len(action_seq) and current_step_agent == perceptor) or wrapped_env.finished

        def get_final_return_object():
            if return_obs:
                return { self.env_to_hashed_obs(wrapped_env, actor, perceptor): 1.0 }
            else:
                return { wrapped_env: 1.0 }

        def is_trust_applied():
            actor_conf = self.agent_confs[actor-1]
            if trust_correction and actor_conf.trust is not None:
                # Find the trust setting for the current agent
                trust_current = [trust_conf for trust_conf in actor_conf.trust if trust_conf.player_id == current_step_agent]
                if len(trust_current) == 1:
                    # Check if the trust correction should be applied at this step (anticipation or one of the following steps)
                    return (trust_current[0].anticipation and anticipation) or action_stepper in trust_current[0].steps
            return False

        def next_state_destroys_empowerment(current_env, next_state):
            follow_up_emp = self.calculate_state_empowerment(next_state, actor, actor, 1)
            if follow_up_emp < EPSILON:
                current_state_emp = self.calculate_state_empowerment(current_env, actor, actor, 1)
                return current_state_emp >= EPSILON
            return False


        if is_final_state():
            return get_final_return_object()

        next_step_agent = current_step_agent % self.player_count + 1
        # Increase the action stepper if this agent was the active one.
        next_action_step = action_stepper + 1 if actor == current_step_agent else action_stepper
        
        # If the current agent is not alive, go straight to the next agent
        if not current_step_agent in find_alive_players(wrapped_env):
            return self.build_distribution(wrapped_env, action_seq, next_action_step, actor, next_step_agent, perceptor, return_obs, anticipation, trust_correction)

        pd_s_nstep = {}
        
        # Find the probability of each action for the current agent
        assumed_policy = self.find_assumed_policy(current_step_agent, wrapped_env) if actor != current_step_agent else {self.action_spaces[current_step_agent-1][action_seq[action_stepper]]: 1.0}
        
        apply_trust = is_trust_applied()

        # Sum up the total probability of all possible actions so we can normalize in the end if needed
        total_prob = 0
        action_to_pd_s = {}
        for action in assumed_policy:
            if assumed_policy[action] == 0:
                continue
            # Get the probability distribution for the next step, given an action
            action_to_pd_s[action] = self.calc_pd_s_a(wrapped_env, current_step_agent, action)

        trust_filtered_actions = list(action_to_pd_s)
        if apply_trust:
            for action in action_to_pd_s:
                next_step_pd_s = action_to_pd_s[action]
                for next_state, next_state_prob in next_step_pd_s.items():
                    # If the actor trusts the current agent, we skip all actions that would reduce the actor's empowerment to zero
                    if next_state_prob > 0.01 and next_state_destroys_empowerment(wrapped_env, next_state):
                        trust_filtered_actions.remove(action)
                        break
        # If all actions led to no empowerment, we should consider them all.
        if len(trust_filtered_actions) == 0:
            trust_filtered_actions = list(action_to_pd_s)

        for action in trust_filtered_actions:
            # Recursively, build the distribution for each possbile follow-up state
            next_step_pd_s = action_to_pd_s[action]
            for next_state, next_state_prob in next_step_pd_s.items():
                child_distribution = self.build_distribution(next_state, action_seq, next_action_step, actor, next_step_agent, perceptor, return_obs, anticipation, trust_correction)
                # Add the follow-up states to the overall distribution of this state p(S_t+n|s_t, a_t^n)
                for next_env in child_distribution:
                    if next_env not in pd_s_nstep:
                        pd_s_nstep[next_env] = 0
                    adjusted_prob = child_distribution[next_env] * assumed_policy[action] * next_state_prob
                    pd_s_nstep[next_env] += adjusted_prob
                    total_prob += adjusted_prob
        
        # Normalize the probabilities. Could not be 1, if we skip one because of trust correction, for instance.
        if total_prob > 0 and total_prob < 1 - EPSILON:
            for state in pd_s_nstep:
                pd_s_nstep[state] /= total_prob
        return pd_s_nstep


    @lru_cache(maxsize=4000)
    def calculate_state_empowerment(self, wrapped_env, actor, perceptor, n_step, trust_correction=False):
        # If the player isn't alive anymore, we assume the empowerment to be zero
        if not wrapped_env.finished and find_player_pos(wrapped_env, actor) is None:
            return 0

        # All possible action combinations of length 'step'
        action_sequences = [tuple(combo) for combo in product(range(len(self.action_spaces[actor-1])), repeat=n_step)]
        # A list of end state probabilities, for each action combination. A list of dictionaries.
        cpd_S_A_nstep = [None] * len(action_sequences)
        for seq_idx, action_seq in enumerate(action_sequences):
            cpd_S_A_nstep[seq_idx] = self.build_distribution(wrapped_env, action_seq, 0, actor, actor, perceptor, return_obs=True, trust_correction=trust_correction)

        # Use the Blahut-Arimoto algorithm to calculate the optimal probability distribution
        pd_a_opt = empowerment_maximization.blahut_arimoto(cpd_S_A_nstep, self.rng)
        # Use the optimal distribution to calculate the mutual information (empowerement)
        empowerment = empowerment_maximization.mutual_information(pd_a_opt, cpd_S_A_nstep)
        return empowerment


    def calculate_expected_empowerments(self, gym_env, current_player, emp_pair, n_step, trust_correction):
        '''
        Calculates the expected empowerment for each action that can be taken from the current state, for one empowerment pair (actor, perceptor).
        '''
        # Find the probability of follow-up states for when the actor is in turn. p(s|s_t, a_t), s = state when actor is in turn
        cpd_s_a_anticipation = {}
        for action_idx, action in enumerate(self.action_spaces[current_player-1]):
            cpd_s_a_anticipation[action] = self.build_distribution(EnvHashWrapper(gym_env.clone()), (action_idx,), 0, current_player, current_player, emp_pair[0], return_obs=False, anticipation=True, trust_correction=trust_correction)
        # Make a flat list of all the reachable states
        reachable_states = set()
        for cpd_s in cpd_s_a_anticipation.values():
            for future_state in cpd_s:
                reachable_states.add(future_state)
        # Save the expected empowerment for each anticipated state
        reachable_state_empowerments = {}
        # Calculate the n-step empowerment for each state that was found earlier
        for state in reachable_states:
            empowerment = self.calculate_state_empowerment(state, emp_pair[0], emp_pair[1], n_step, trust_correction=trust_correction)
            reachable_state_empowerments[state] = empowerment
        # Calculate the expected empowerment for each action that can be taken from the current state
        expected_empowerments = {}
        for a in self.action_spaces[current_player-1]:
            for state, state_probability in cpd_s_a_anticipation[a].items():
                if not a in expected_empowerments:
                    expected_empowerments[a] = 0
                expected_empowerments[a] += state_probability * reachable_state_empowerments[state]
        
        return expected_empowerments

    
    def env_to_hashed_obs(self, wrapped_env, actor, perceptor):
        if wrapped_env.finished:
            # If the actor is in the winning team, all their actions lead to maximum empowerment
            if actor == wrapped_env.winner:
                hashed_obs = self.rng.integers(100, 4000000000)
                return hashed_obs
            # But if the actor loses future actions lead to a minimum empowerment
            elif wrapped_env.winner > 0:
                hashed_obs = wrapped_env.winner
                return hashed_obs
        latest_obs = wrapped_env.player_last_obs(perceptor)
        player_pos = find_player_pos(wrapped_env, perceptor)
        # If the player was not found, we assume the player is dead. In that case, the hashed observation is the player's id
        hashed_obs = hash_obs(latest_obs, player_pos) if player_pos is not None else hash(actor)
        return hashed_obs
