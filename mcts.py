from math import log, sqrt
import random
import env_util
import global_configuration
import numpy as np


# Scores this close to each other will be considered equal.
SCORE_EPSILON = 0.005
# How much to weigh exploration compared to exploitation.
EXPLORATION_WEIGHT = 4*sqrt(2)
# Do we just skip the other agents' actions or not?
SKIP_OPPONENTS = False
# How steep should the sigmoid function be?
SIGMOID_K = 2

class MCTS:
    def __init__(self, env, actor, action_spaces):
        self.root = Node()
        self.env = env
        self.actor = actor
        self.action_spaces = action_spaces
        self.iteration_count = 0
        # Variance and standard deviation of all simulation results
        self._variance = 0.0
        self.std_dev = 0.0
        self.avg_raw_result = 0.0
        # Warning: this is updated only in the beginning and cannot be trusted furthermore.
        self._raw_results = np.array([], np.int32)

    def iterate(self):
        rewards = [0 for _ in range(self.env.player_count)]
        clone_env = self.env.clone()
        self.root.iterate(self, clone_env, self.actor, self.action_spaces, iteration_rewards=rewards, is_root=True)

    def normalize_result(self, result_raw):
        return result_raw
        # # If this is the first result, it's the average.
        # if self.iteration_count == 0 or self.std_dev == 0:
        #     return 0.5

        # def sigmoid(val):
        #     return 1 / (1 + np.exp(-val*SIGMOID_K))

        # # How many standard deviations from the mean is this result?
        # stds_from_mean = (result_raw - self.avg_raw_result) / self.std_dev
        # return sigmoid(stds_from_mean)

    def update_averages(self, raw_result):
        self.avg_raw_result = (self.avg_raw_result * self.iteration_count + raw_result) / (self.iteration_count + 1)
        self.iteration_count += 1
        if self.iteration_count < 10:
            self._raw_results = np.append(self._raw_results, [raw_result])
            self._variance = np.var(self._raw_results)
            self.std_dev = np.std(self._raw_results)
            return

        # https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
        self._variance = (self.iteration_count - 2) / (self.iteration_count - 1) * self._variance + (1 / self.iteration_count) * (raw_result - self.avg_raw_result)**2
        self.std_dev = sqrt(self._variance)


class Node:
    def __init__(self):
        self.children = []
        self.visits = 0
        self.avg_score = 0


    def iterate(self, mcts, env, actor, action_spaces, iteration_rewards, is_root=False):
        # If this is a leaf node, simulate the game
        if self.visits == 0 and not is_root:
            iteration_result = self.simulate(mcts, env, actor, action_spaces, iteration_rewards)
        else:
            # Select children until a leaf node is reached (leaf is any node without children)
            if len(self.children) < len(action_spaces[actor-1]):
                selected_child_i = self.expand(env, actor)
            else:
                selected_child_i = self.select_child()
            
            selected_child = self.children[selected_child_i]
            action = action_spaces[actor-1][selected_child_i]
            obs, reward, done, info = env.step(env_util.build_action(action, len(action_spaces), actor))
            for reward_i, reward in enumerate(reward):
                iteration_rewards[reward_i] += reward

            if global_configuration.visualise_all:
                env.render(mode='human', observer='global')
            
            if done:
                iteration_result = mcts.normalize_result(iteration_rewards[actor-1])
                selected_child.update_with_result(iteration_result)
            else:
                if isinstance(selected_child, OpponentNode):
                    next_agent_id = actor % env.player_count + 1 if not SKIP_OPPONENTS else actor
                    iteration_result = selected_child.iterate(mcts, env, actor, action_spaces, iteration_rewards, is_root=False, player_in_turn=next_agent_id)
                else:
                    iteration_result = selected_child.iterate(mcts, env, actor, action_spaces, iteration_rewards)

        self.update_with_result(iteration_result)
        return iteration_result


    def expand(self, env, actor):
        next_agent_id = actor % env.player_count + 1 if not SKIP_OPPONENTS else actor
        self.children.append(Node() if next_agent_id == actor else OpponentNode())
        selected_child_i = len(self.children) - 1
        return selected_child_i


    def update_with_result(self, score):
        self.visits += 1
        self.avg_score = (self.avg_score * (self.visits - 1) + score) / self.visits


    def select_child(self):
        assert(self.visits > 0)
        log_parent_visits = log(self.visits)

        # Select the best child
        def uct(child):
            # Exploration-exploitation trade-off
            exploitation = child.avg_score
            exploration = EXPLORATION_WEIGHT * sqrt(log_parent_visits / child.visits)
            return exploitation + exploration

        best_child = max(self.children, key=uct)
        return self.children.index(best_child)


    def best_child_idx(self):
        best_children_idx = []
        best_children_score = 0.0
        for child_i, child in enumerate(self.children):
            child_score = child.avg_score
            if child_score > best_children_score + SCORE_EPSILON:
                best_children_score = child_score
                best_children_idx = [child_i]
            elif abs(child_score - best_children_score) < SCORE_EPSILON:
                best_children_idx.append(child_i)
        return random.choice(best_children_idx)


    # Simulate the game until the end with random moves.
    def simulate(self, mcts, env, player_in_turn, action_spaces, iteration_rewards):
        current_agent = player_in_turn
        done = False
        result = 0
        # Simulate the game
        while not done:
            # Select a random action
            action = random.choice(action_spaces[current_agent-1])
            obs, reward, done, info = env.step(env_util.build_action(action, len(action_spaces), current_agent))
            for i in range(env.player_count):
                iteration_rewards[i] += reward[i]
            if global_configuration.visualise_all:
                env.render(mode='human', observer='global')
            if not SKIP_OPPONENTS:
                current_agent = current_agent % env.player_count + 1
        result = iteration_rewards[player_in_turn-1]
        normalized_res = mcts.normalize_result(result)
        mcts.update_averages(result)
        return normalized_res


class OpponentNode(Node):
    def iterate(self, mcts, env, actor, action_spaces, iteration_rewards, is_root=False, *, player_in_turn):
        '''
        Opponent nodes are different in that they do not try to find the best actions.
        Instead, they choose actions uniformly.
        Iterate function therefore returns the result of a randomly chosen child node.
        '''
        if not self.children:
            next_agent_id = player_in_turn % env.player_count + 1
            if (next_agent_id == actor):
                self.children = [Node() for _ in action_spaces[player_in_turn-1]]
            else:
                self.children = [OpponentNode() for _ in action_spaces[player_in_turn-1]]

        selected_child_i = random.choice(range(len(self.children)))
        selected_child = self.children[selected_child_i]
        obs, rew, done, info = env.step(env_util.build_action(action_spaces[player_in_turn-1][selected_child_i], len(action_spaces), player_in_turn))
        if global_configuration.visualise_all:
            env.render(mode='human', observer='global')

        for i in range(env.player_count):
            iteration_rewards[i] += rew[i]
        
        if done:
            iter_result = mcts.normalize_result(iteration_rewards[player_in_turn-1])
            mcts.update_averages(iteration_rewards[player_in_turn-1])
        else:
            if isinstance(selected_child, OpponentNode):
                next_agent_id = player_in_turn % env.player_count + 1
                iter_result = selected_child.iterate(mcts, env, actor, action_spaces, iteration_rewards, is_root=False, player_in_turn=next_agent_id)
            else:
                iter_result = selected_child.iterate(mcts, env, actor, action_spaces, iteration_rewards)
        
        self.visits += 1
        self.avg_score = (self.avg_score * (self.visits - 1) + iter_result) / self.visits
        return iter_result
