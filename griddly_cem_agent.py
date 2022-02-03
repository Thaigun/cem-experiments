from itertools import product
import empowerment_maximization
import numpy as np
from functools import lru_cache
import random
from collections import namedtuple


EPSILON = 1e-5


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
    # If health is not an object variable, return None
    if 'health' not in player_variables[0]:
        return None
    return player_variables[0]['health']


def find_alive_players(wrapped_env):
    return [player_id for player_id in range(1, wrapped_env.player_count + 1) if wrapped_env.get_available_actions(player_id)]


# Returns the player_id of the player who has won.
# TODO: What to return if the game has ended but there is no winner?
def find_winner(env, info):
    if 'PlayerResults' in info:
        for plr, status in info['PlayerResults'].items():
            if status == 'Win':
                return int(plr)
    return -1


class EnvHashWrapper():
    # This is a bit dangerous, but the env that is given to the constructor must not be stepped.
    # If we can find a way to not lose too much performance by cloning it here in the constructor, that would be perfect.
    def __init__(self, env):
        if not env._is_clone:
            raise ValueError('The environment given to EnvHashWrapper must be a clone of an original environment.')
        self._env = env
        self.hash = None

    def __hash__(self) -> int:
        return self.get_hash()

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, EnvHashWrapper) and self.__hash__() == __o.__hash__()

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
    def __init__(self, winner):
        self.winner = int(winner)

    def __hash__(self) -> int:
        return self.winner

    def __eq__(self, __o: object) -> bool:
        try:
            return self.winner == __o.winner
        except:
            return False


class CEM():
    def __init__(self, env, agent_confs, seed=None, samples=1):
        self.samples = samples
        self.player_count = env.player_count
        self.agent_confs = agent_confs

        # List all possible actions in the game
        self.action_spaces = [[] for _ in range(self.player_count)] 
        # Include the idling action
        for player_i in range(self.player_count):
            player_i_actions = self.agent_confs[player_i]['Actions']
            if 'idle' in player_i_actions:
                self.action_spaces[player_i].append((0,0))
            for action_type_index, action_name in enumerate(env.action_names):
                if action_name in player_i_actions:
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

        for emp_pair in self.agent_confs[player_id-1]['EmpowermentPairs']:
            expected_empowerments = self.calculate_expected_empowerments(env, player_id, (emp_pair['Actor'], emp_pair['Perceptor']), n_step, True)
            expected_empowerments_per_pair.append(expected_empowerments)

        best_actions = []
        best_rewards = []
        # Find the action index that yields the highest expected empowerment
        for a in self.action_spaces[player_id-1]:
            action_reward = 0
            for emp_pair_i, agent_emp_conf in enumerate(self.agent_confs[player_id-1]['EmpowermentPairs']):
                action_reward += agent_emp_conf['Weight'] * expected_empowerments_per_pair[emp_pair_i][a]
            if not len(best_rewards) or action_reward >= max(best_rewards) + EPSILON:
                best_rewards = [action_reward]
                best_actions = [a]
            elif action_reward > max(best_rewards) - EPSILON and action_reward < max(best_rewards) + EPSILON:
                best_rewards.append(action_reward)
                best_actions.append(a)
        
        return random.choice(best_actions)


    # Builds the action that can be passed to the Griddly environment
    def build_action(self, action, player_id):
        built_action = [[0,0] for _ in range(self.player_count)]
        built_action[player_id-1] = list(action)
        return built_action


    @lru_cache(maxsize=5000)
    def calc_pd_s_a(self, env, player_id, action):
        # Build it lazily
        player_health = find_player_health(env.get_state(), player_id)
        if player_health is not None:
            health_ratio = player_health / self.agent_confs[player_id-1]['MaxHealth']
        else:
            health_ratio = 1.0
        # if the player is dead, the empowerment is 0
        if health_ratio == 0:
            return {env: 1.0}

        result = {}
        for _ in range(self.samples):
            clone_env, obs, rew, env_done, info = env.step(self.build_action(action, player_id))
            if (env_done):
                game_winner = find_winner(clone_env, info)
                next_env = GameEndState(game_winner)
            else:
                next_env = clone_env
            if next_env not in result:
                result[next_env] = 0
            result[next_env] += 1.0 / self.samples

        # Adjust for health-performance consistency
        if health_ratio < 1 - EPSILON:
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
        # TODO: Verify this is correct
        transitions[env] = 1 - health_ratio + health_ratio * transitions[env]


    def find_assumed_policy(self, agent_id, wrapped_env):
        current_agent_conf = self.agent_confs[agent_id-1]
        # Find the policy of the agent in turn
        return current_agent_conf['AssumedPolicy'](wrapped_env._env , self, agent_id)


    #@lru_cache(maxsize=2000)
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
        # If this is the last step, end the recursion
        if (action_stepper == len(action_seq) and current_step_agent == perceptor) or isinstance(wrapped_env, GameEndState):
            # if return_obs is True, then we return the observation instead of the state
            return {self.env_to_hashed_obs(wrapped_env, actor, perceptor): 1.0} if return_obs else {wrapped_env: 1.0}

        next_step_agent = current_step_agent % self.player_count + 1
        # Also, increase the action stepper if this agent was the active one.
        next_action_step = action_stepper + 1 if actor == current_step_agent else action_stepper
        
        # If the current agent is not alive, go straight to the next agent
        if not current_step_agent in find_alive_players(wrapped_env):
            return self.build_distribution(wrapped_env, action_seq, next_action_step, actor, next_step_agent, perceptor, return_obs, anticipation, trust_correction)

        actor_conf = self.agent_confs[actor-1]
        pd_s_nstep = {}
        
        # Find the probability of each acttion for the current agent
        assumed_policy = self.find_assumed_policy(current_step_agent, wrapped_env) if actor != current_step_agent else {self.action_spaces[current_step_agent-1][action_seq[action_stepper]]: 1.0}

        # Sum up the total probability of all possible actions so we can normalize in the end if needed
        total_prob = 0
        # TODO: Should this be the action index or actual action?
        for action in assumed_policy:
            if assumed_policy[action] == 0:
                continue
            # Get the probability distribution for the next step, given an action
            next_step_pd_s = self.calc_pd_s_a(wrapped_env, current_step_agent, action)

            # Recursively, build the distribution for each possbile follow-up state
            for next_state, next_state_prob in next_step_pd_s.items():
                # If the actor trusts the current agent, we skip all actions that would reduce the actor's empowerment to zero
                # TODO: It feels like the potential trust correction step could be separated into a separate function
                if trust_correction and 'Trust' in actor_conf and next_state_prob > 0.01:
                    # Find the trust setting for the current agent
                    trust_current = [trust_conf for trust_conf in actor_conf['Trust'] if trust_conf['PlayerId'] == current_step_agent]
                    if len(trust_current) == 1:
                        # Check if the trust correction should be applied at this step (anticipation or one of the following steps)
                        # TODO: Check that the trust steps config makes sense.
                        if (trust_current[0]['Anticipation'] and anticipation) or action_stepper in trust_current[0]['Steps']:
                            follow_up_emp = self.calculate_state_empowerment(next_state, actor, actor, 1)
                            if follow_up_emp < EPSILON:
                                current_state_emp = self.calculate_state_empowerment(wrapped_env, actor, actor, 1)
                                if current_state_emp != 0:
                                    break

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


    @lru_cache(maxsize=8000)
    def calculate_state_empowerment(self, wrapped_env, actor, perceptor, n_step, trust_correction=False):
        # If the player isn't alive anymore, we assume the empowerment to be zero
        if not isinstance(wrapped_env, GameEndState) and find_player_pos(wrapped_env, actor) is None:
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
            empowerment = self.calculate_state_empowerment(state, emp_pair[0], emp_pair[1], n_step)
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
        if isinstance(wrapped_env, GameEndState):
            # If the actor is in the winning team, all their actions lead to maximum empowerment
            if actor == wrapped_env.winner:
                hashed_obs = self.rng.integers(100, 4000000000)
            # But if the actor loses futura actions lead to a minimum empowerment
            else:
                hashed_obs = wrapped_env.winner
        else:
            latest_obs = wrapped_env.player_last_obs(perceptor)
            player_pos = find_player_pos(wrapped_env, perceptor)
            # If the player was not found, we assume the player is dead. In that case, the hashed observation is the player's id
            hashed_obs = hash_obs(latest_obs, player_pos) if player_pos is not None else hash(actor)
        return hashed_obs
