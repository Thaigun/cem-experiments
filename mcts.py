from math import log, sqrt
import random
import env_util
import configuration
import numpy as np


# Scores this close to each other will be considered equal.
SCORE_EPSILON = 0.1
# How much to weigh exploration compared to exploitation.
EXPLORATION_WEIGHT = 4
# How quickly we want to announce a score to be good or bad. Higher the value, the more unconfident we are.
SCORE_UNCONFIDENCE = 12.0
# Do we just skip the other agents' actions or not?
SKIP_OPPONENTS = True


class MCTS:
    def __init__(self, env, actor, action_spaces):
        self.root = Node()
        self.env = env.clone()
        self.actor = actor
        self.action_spaces = action_spaces
        # Variance and standard deviation of all simulation results
        self._variance = 0.0
        self.std_dev = 0.0

    def iterate(self):
        previous_avg_score = self.avg_score
        result = self.root.iterate(self, self.env, self.actor, self.actor, self.action_spaces, is_root=True)
        # Update the variance incrementally
        self._variance = (self.iteration_count - 2) / (self.iteration_count - 1) * self._variance + (1 / self.iteration_count) * (result - previous_avg_score)**2
        self.std_dev = sqrt(self._variance)

    @property
    def avg_score(self):
        return self.root.avg_score

    @property
    def iteration_count(self):
        return self.root.visits


class Node:
    def __init__(self):
        self.children = []
        self.visits = 0
        self.avg_score = 0


    def iterate(self, mcts, env, actor, player_in_turn, action_spaces, max_sim_steps=10000, is_root=False):
        # If this is a leaf node, simulate the game
        if self.visits == 0 and not is_root:
            round_result = self.simulate(env, player_in_turn, action_spaces, max_sim_steps)
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
            # TODO: What if the game ends here?
            obs, reward, done, info = env.step(env_util.build_action(action, len(action_spaces), player_in_turn))
            if configuration.visualise_all:
                env.render(mode='human', observer='global')
            
            if done:
                selected_child.visits += 1
                winner = env_util.find_winner(info)
                if winner == -1:
                    winner = actor
                if winner == actor:
                    round_result = 1
                else:
                    round_result = 2*max_sim_steps
            # Update the average score and backpropagate the result
            else:
                next_agent_id = player_in_turn % env.player_count + 1 if not SKIP_OPPONENTS else actor
                round_result = selected_child.iterate(env, actor, next_agent_id, action_spaces, max_sim_steps-1, is_root=False)

        self.visits += 1
        self.avg_score = (self.avg_score * (self.visits - 1) + round_result) / self.visits
        return round_result + 1


    def select_child(self):
        assert(self.visits > 0)
        log_parent_visits = log(self.visits)
        children_scores_arr = np.array([child.avg_score for child in self.children])
        score_average = children_scores_arr.mean()
        score_stdev = children_scores_arr.std()

        def sigmoid(val):
            return 1 / (1 + np.exp(-val))

        # Select the best child
        def child_score(child):
            # Exploration-exploitation trade-off
            #score_confidence = child.visits / (child.visits + SCORE_UNCONFIDENCE)
            stds_from_mean = (child.avg_score - score_average) / score_stdev
            #exploitation = sigmoid(-stds_from_mean * score_confidence)
            exploitation = sigmoid(-stds_from_mean)
            exploration = EXPLORATION_WEIGHT * sqrt(log_parent_visits / child.visits)
            return exploitation + exploration

        best_child = max(self.children, key=child_score)
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
    # TODO: the steps taken to get down to this level should be counted in the score.
    def simulate(self, env, player_in_turn, action_spaces, max_sim_steps):
        step_count = 0
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
        return result


class OpponentNode(Node):
    def iterate(self, env, actor, player_in_turn, action_spaces, total_iter_count, max_sim_steps=10000, is_root=False):
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
            round_res = self.children[selected_child_i].iterate(env, actor, player_in_turn % env.player_count + 1, action_spaces, total_iter_count, max_sim_steps-1, is_root=False)
        
        self.visits += 1
        self.avg_score = (self.avg_score * (self.visits - 1) + round_res) / self.visits
        return round_res + 1
