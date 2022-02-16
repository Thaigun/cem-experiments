from math import log, sqrt
import random
import env_util
import configuration
import numpy as np


# Scores this close to each other will be considered equal.
SCORE_EPSILON = 0.005
# How much to weigh exploration compared to exploitation.
EXPLORATION_WEIGHT = sqrt(1.5)
# Do we just skip the other agents' actions or not?
SKIP_OPPONENTS = True
# How steep should the sigmoid function be?
SIGMOID_K = 2

class MCTS:
    def __init__(self, env, actor, action_spaces, max_sim_steps=20000):
        self.root = Node()
        self.env = env
        self.actor = actor
        self.action_spaces = action_spaces
        self.iteration_count = 0
        self.max_sim_steps = max_sim_steps
        # Variance and standard deviation of all simulation results
        self._variance = 0.0
        self.std_dev = 0.0
        self.avg_raw_result = 0.0
        # Warning: this is updated only in the beginning and cannot be trusted furthermore.
        self._raw_results = np.array([], np.int32)

    def iterate(self):
        self.root.iterate(self, self.env.clone(), self.actor, self.actor, self.action_spaces, depth=0, max_sim_steps=self.max_sim_steps, is_root=True)

    def normalize_result(self, result_raw):
        # If this is the first result, it's the average.
        if self.iteration_count == 0 or self.std_dev == 0:
            return 0.5

        def sigmoid(val):
            return 1 / (1 + np.exp(-val*SIGMOID_K))

        # How many standard deviations from the mean is this result?
        stds_from_mean = (result_raw - self.avg_raw_result) / self.std_dev
        return sigmoid(-stds_from_mean)

    def update_averages(self, raw_result):
        self.avg_raw_result = (self.avg_raw_result * self.iteration_count + raw_result) / (self.iteration_count + 1)
        self.iteration_count += 1
        if self.iteration_count < 10:
            self._raw_results = np.append(self._raw_results, [raw_result]) # TODO: This is only needed during the first few iterations
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


    def iterate(self, mcts, env, actor, player_in_turn, action_spaces, depth, max_sim_steps=10000, is_root=False):
        # If this is a leaf node, simulate the game
        if self.visits == 0 and not is_root:
            round_result = self.simulate(mcts, env, player_in_turn, action_spaces, depth, max_sim_steps)
        else:
            # Select children until a leaf node is reached (leaf is any node without children)
            if len(self.children) < len(action_spaces[player_in_turn-1]):
                next_agent_id = player_in_turn % env.player_count + 1 if not SKIP_OPPONENTS else actor
                self.children.append(Node() if next_agent_id == actor else OpponentNode())
                selected_child_i = len(self.children) - 1
            else:
                # Select the best child
                selected_child_i = self.select_child()
            
            selected_child = self.children[selected_child_i]
            action = action_spaces[player_in_turn-1][selected_child_i]
            obs, reward, done, info = env.step(env_util.build_action(action, len(action_spaces), player_in_turn))
            if configuration.visualise_all:
                env.render(mode='human', observer='global')
            
            if done:
                winner = env_util.find_winner(info)
                if winner == -1:
                    winner = actor
                if winner == actor:
                    round_result = mcts.normalize_result(depth+1)
                else:
                    round_result = 0
                selected_child.update_with_result(round_result)
            # Update the average score and backpropagate the result
            else:
                next_agent_id = player_in_turn % env.player_count + 1 if not SKIP_OPPONENTS else actor
                round_result = selected_child.iterate(mcts, env, actor, next_agent_id, action_spaces, depth+1, max_sim_steps, is_root=False)

        self.update_with_result(round_result)
        return round_result


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
    def simulate(self, mcts, env, player_in_turn, action_spaces, depth, max_sim_steps):
        step_count = depth
        current_agent = player_in_turn
        result = max_sim_steps
        # Simulate the game
        for _ in range(max_sim_steps):
            # Select a random action
            action = random.choice(action_spaces[current_agent-1])
            obs, reward, done, info = env.step(env_util.build_action(action, len(action_spaces), current_agent))
            if configuration.visualise_all:
                env.render(mode='human', observer='global')
            step_count += 1
            if not SKIP_OPPONENTS:
                current_agent = current_agent % env.player_count + 1
            if done:
                winner = env_util.find_winner(info)
                # If the environment is done but there is no winner, we assume it was a good result
                if winner == -1:
                    winner = player_in_turn
                if winner == player_in_turn:
                    result = step_count
                else:
                    result = 2*step_count
                break
        normalized_res = mcts.normalize_result(result)
        mcts.update_averages(result)
        return normalized_res


class OpponentNode(Node):
    def iterate(self, mcts, env, actor, player_in_turn, action_spaces, depth, max_sim_steps=10000, is_root=False):
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
        obs, rew, done, info = env.step(env_util.build_action(action_spaces[player_in_turn-1][selected_child_i], len(action_spaces), player_in_turn))
        if configuration.visualise_all:
            env.render(mode='human', observer='global')
        if done:
            winner = env_util.find_winner(info)
            if winner == -1:
                winner = actor
            if winner == actor:
                round_res = 1
            else:
                round_res = 2*max_sim_steps
        else:
            round_res = self.children[selected_child_i].iterate(mcts, env, actor, player_in_turn % env.player_count + 1, action_spaces, depth=depth+1, max_sim_steps=max_sim_steps-1, is_root=False)
        
        self.visits += 1
        self.avg_score = (self.avg_score * (self.visits - 1) + round_res) / self.visits
        return round_res + 1
