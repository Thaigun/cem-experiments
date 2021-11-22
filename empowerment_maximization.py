"""
Module containing classes related to empowerment maximisation (EM).
Agents under EM act to increase their influence on the environment/
future sensor states via their actions.

By Christian Guckelsberger
"""
import logging
import gym
import numpy as np
from griddly import gd

MAX_FLOAT = MAX_FLOAT = np.finfo(np.float32).max
EPSILON_1 = 1e-5
EPSILON_2 = 1e-9
EPSILON_3 = 1e-3

def marginalise(pd_y, cpd_xy):
    """
    Marginalise p(X|Y) with respect to p(Y). We get a new distribution p(X).
    """
    pd_x = {}
    for y, p_y in pd_y.items():
        for x, p_xy in cpd_xy[y].items():
            p_x = pd_x.get(x, 0.0)
            pd_x[x] = p_x + (p_xy * p_y)

    return pd_x

def plogpd_q(p, q): #pylint: disable=C0103
    """
    Natural logarithm of p divided by q - checking for extreme values to match convention
    """
    if p < EPSILON_2:
        return 0.0
    elif q < EPSILON_2:
        return MAX_FLOAT #huge value
    else:
        return p * np.log(p/q) #do not need base 2 here

def blahut_arimoto(cpd_xy, rand_gen):
    """
    The Blahut-Arimoto algorithm to calculate the channel
    capacity of conditional probability distribution p(X|Y)

    Arguments:
        cond_prob_xy -- scipy sparse matrix, representing information-theoretic channel from Y to X
        rand -- a random number generator; allows to hand over a generator with a fixed seed for reproducibility
    Returns:
        A numpy array, representing p(Y) which maximises the capacity of channel p(X|Y)
    """

    # Initialise uniform random distribution with some noise to break the symmetry
    nY = len(cpd_xy)
    pd_y = {}
    pd_y_sum = 0.0
    for y in range(nY):
        rand = rand_gen.uniform(0.9, 1)
        pd_y[y] = rand
        pd_y_sum += rand

    for y in range(nY):
        pd_y[y] = pd_y[y]/pd_y_sum

    # Do until P*(Y) converges
    converged = False
    while not converged:
        # Marginalise
        pd_x = marginalise(pd_y, cpd_xy)

        pd_y_sum = 0.0
        pd_y_new = {}
        for y, p_y in pd_y.items():
            exponent = 0.0
            for x, p_xy in cpd_xy[y].items():
                if pd_x[x] > EPSILON_1 and p_xy > EPSILON_1:
                    exponent += plogpd_q(p_xy, pd_x[x])
            temp = np.exp(exponent) * p_y
            pd_y_new[y] = temp
            pd_y_sum += temp

        # Normalise
        for y in range(nY):
            pd_y_new[y] /= pd_y_sum

        # Check for convergence
        converged = True
        for y in range(nY):
            if abs(pd_y[y]-pd_y_new[y]) > EPSILON_3:
                converged = False
                break

        # Store the new values persistently for the next loop in pd_y
        pd_y = pd_y_new

    return pd_y

def mutual_information(pd_y, cpd_xy):
    """
    Mutual information I(X;Y) making use of conditional distribution p(Y|X)
    NOT TO BE MISTAKEN WITH CONDITIONAL MUTUAL INFORMATION!

    Uses the Kullback-Leibler divergence to calclulate mutual information

    Arguments:
        pd_y: probability distribution of size |Y|.
        cond_prob_xy: conditional probability distribution of size |X|x|Y|
    """
    # Marginalise
    pd_x = marginalise(pd_y, cpd_xy)

    mi = 0.0
    for y, p_y in pd_y.items():
        kl_divergence = 0.0
        for x, p_xy in cpd_xy[y].items():
            if pd_x[x] > 0:
                log_prob = np.log2(p_xy)-np.log2(pd_x[x])
                kl_divergence += p_xy * log_prob
        mi += p_y * kl_divergence

    return mi

class EMVanillaNStepAgent():
    """
    Vanilla n-step empowerment baseline agent
    Calculates n-step empowerment for all potential follow-up states
    Then selects action with maximum expected empowerment
    """

    def __init__(self, n_step, samples, env, one_step_anticipation=True, player_id=2):
        """
        Initialise this agent
        Args:
            nstep -- number of steps lookahead into the future
            one_step_anticipation -- by default, the agent evaluates empowerment of states
                        one step ahead, and then moves into state with maximum empowerment.
                        Deactivating this means that only the present state is being evaluated.
        """

        self._observation_space = env.observation_space
        self._action_space = [(0, 0)] # Include the idling action
        for action_type_index, action_name in enumerate(env.action_names):
            for action_id in range(1, env.num_action_ids[action_name]):
                self._action_space.append((action_type_index, action_id))

        self._player_id = player_id
        self._n_step = n_step
        self._samples = samples
        self._one_step_anticipation = one_step_anticipation
        self._train_after = 0
        self.np_random = np.random.default_rng()

        self._logger = logging.getLogger(__name__)
        self._logger.info("Vanilla %d-step empowerment agent", n_step)

    def observe(self, pre_observation, action, reward, post_observation, done):
        if done:
            self._logger.info("Completed episode.")

    def build_action(self, action):
        if (self._player_id == 1):
            return [action, [0,0]]
        elif (self._player_id == 2):
            return [[0,0], action]
        return [[0,0],[0,0]]

    def sample(self, env, action_idx, samples, hash_decode):
        """
        Sample action transitions
        Args:
            env: previous env
            action: action to perform
            samples: number of samples ot perform
            hash_decode -- a dictionary mapping state hashes to state values
        """

        # Empirical point estimate
        pd_s = {}

        p_sample = 1.0 / samples
        for _ in range(0, samples):
            new_env = env.clone()
            new_env.step(self.build_action(list(self._action_space[action_idx])))
            new_state = new_env.get_state()
            s_new = new_state['Hash']
            hash_decode[s_new] = new_env
            p_s = pd_s.get(s_new, 0.0)
            pd_s[s_new] = p_s + p_sample

        return (pd_s, hash_decode)

    def forward(self, env, n_step, hash_decode):
        """
        Expands forward model for n step lookahead into distribution
        p(S_{t+n}|S_t, A^n_t = (A_t,A_t+1,...A_{t+n-1}))
        Basically breadth-first tree expansion
        Args:
            state -- the state to query
            n_step -- the lookahead; allows to use this method for forward calls of arbitrary
                      length
            hash_decode -- a dictionary mapping state hashes to state values

        Returns:
            a 2-dim sparse array, mapping from all n-step action sequences to follow
            up-states n steps away i.e. p(S_{t+n}|A^n_t, s_t)
        """
        nA = len(self._action_space)
        nASeq = pow(nA, n_step)
        cpd_s_a_nstep = {key: None for key in range(0, nASeq)}

        # Start from given state
        # Keep record of state hashes
        s_init = env.get_state()['Hash']
        hash_decode[s_init] = env
        cpd_s_a_nstep[0] = {s_init : 1.0}
        for n in range(0, n_step):
            # Delta specifies the index shift between trajectories on this level.
            # The further we go down the tree, the more trajectories there are, the
            # smaller the delta.
            delta = pow(nA, n_step-n-1)
            # For each existing trajectory, pick up previously expanded states and expand
            # for each available action
            for aSeq in range(0, nASeq, delta*nA):
                pd_s = cpd_s_a_nstep[aSeq]
                for a in range(nA-1, -1, -1):
                    # Expand and summarise double states
                    pd_s_expanded = {}
                    for s, p_s in pd_s.items():
                        state = hash_decode[s]

                        (pd_sNew, hash_decode) = self.sample(state, a, self._samples, hash_decode)

                        for s_new, p_s_new in pd_sNew.items():
                            # get() returns 0.0 as default probability if s_new not present
                            p_Agg = pd_s_expanded.get(s_new, 0.0)
                            pd_s_expanded[s_new] = p_Agg + (p_s_new * p_s)

                    # Then save new distribution in new index
                    cpd_s_a_nstep[aSeq + a*delta] = pd_s_expanded

        return (cpd_s_a_nstep, hash_decode)

    def _determine_states(self, env):
        """Determine the states to calculate empowerment on.

        Description:
            Two options:
            1.1)  For all actions a_t: Query forward model and get potential follow-up states:
                  p(S_{t+1}|A_t,s_t) for all actions A_t: |S|x|A| matrix. This is independent
                  of a policy; we look at all possible actions equally.
            1.2)  Calculate empowerment only on the present state. This is particularly useful
                  if we want to build a map of the rewards in a environment, without really
                  acting on them.

        Args:
            observation -- the latest observation

        Returns p(s|A), hash_decode for the observation
        """
        cpd_s_A = {}
        hash_decode = {}
        if self._one_step_anticipation:
            (cpd_s_A, hash_decode) = self.forward(env, 1, hash_decode)
        else:
            s_init = env.get_state()['Hash']
            hash_decode[s_init] = env
            action_count = len(self._action_space)
            # for action count, add initial value 1.0
            for a in range(action_count):
                cpd_s_A[a] = {s_init: 1.0}

        return cpd_s_A, hash_decode

    def _initialize_empowerment(self, cpd_s_A, hash_decode):
        """Initialize empowerment for all follow-up states.

        Description:
            (Separately for each immediate follow-up state s_{t+1} caused by a_t)

        Returns e_s
        """
        e_s = {} # Empowerment in state s
        for a in range(0, len(self._action_space)):
            for sNew in cpd_s_A[a]:
                # 2.1) Look up if empowerment for this state has been calculated before (dictionary)
                if sNew in e_s:
                    continue

                # 2.2) If not, calculate n-step predictive model p(S_{t+n+1}|A_{t+1},s_{t+1})
                state_new = hash_decode[sNew]
                (cpd_s_a_nstep, hash_decode) = self.forward(state_new, self._n_step, hash_decode)

                # 2.3) Calculate n-step empowerment via BA on predictive model
                pd_a_opt = blahut_arimoto(cpd_s_a_nstep, self.np_random)
                # Gives us p(a^n_{t+1}) which maximises p(S_{t+n+1}|A_{t+1},s_{t+1})
                empowerment = mutual_information(pd_a_opt, cpd_s_a_nstep)

                # 2.4) Store empowerment for s_t+1 in separate dictionary so no empowerment
                #      calculations are repeated
                e_s[sNew] = empowerment

        return e_s

    def _calculate_expected_empowerment(self, e_s, cpd_s_A):
        """For all actions a_t: Calculate expected empowerment.

        Returns e_A
        """
        e_A = np.zeros(len(self._action_space))
        for a in range(0, len(self._action_space)):
            for s, p_s in cpd_s_A[a].items():
                e_A[a] += p_s * e_s[s]

        return e_A

    def _greedy_action(self, e_A):
        """Greedily pick action with highest expected empowerment.

        Description:
            Round all expectations to number of digits to epsilon-precision: if e.g.
            epsilon=1e-5, round to 5 digits. If differences in empowerment < epsilon,
            pick one at random from overall set
            If not, pick one action at random from the subset of maximising actions

        Return the next action to take
        """
        aNew = -1
        e_A = e_A.round(decimals=len(str(EPSILON_1)))
        if e_A.max()-e_A.min() < EPSILON_1:
            aNew = self.np_random.integers(len(self._action_space))
        else:
            aMax = np.argwhere(e_A == np.amax(e_A)).flatten()
            aNew = self.np_random.choice(aMax)

        return aNew

    def act(self, env):
        """
        Return the next action the agent should take.
        Args:
            env -- the most recent environment state

        Returns:
            a scalar action, randomly picked
        """
        cpd_s_A, hash_decode = self._determine_states(env)
        e_s = self._initialize_empowerment(cpd_s_A, hash_decode)
        e_A = self._calculate_expected_empowerment(e_s, cpd_s_A)
        #print(e_A[0])
        action = self._greedy_action(e_A)

        return list(self._action_space[action])