"""
Module containing classes related to empowerment maximisation (EM).
Agents under EM act to increase their influence on the environment/
future sensor states via their actions.

By Christian Guckelsberger
"""
import numpy as np

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
