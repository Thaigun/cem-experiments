from math import log, sqrt
import random
import env_util
import configuration


SELECTION_EPSILON = 0.01
SCORE_EPSILON = 0.01
EXPLORATION_FACTOR = 500


class Node:
    def __init__(self):
        self.children = []
        self.visits = 0
        self.avg_score = 0


    def iterate(self, env, actor, player_in_turn, action_spaces, total_iter_count=None, max_sim_steps=10000, is_root=False):
        if total_iter_count == None and not is_root:
            raise Exception('total_iter_count must be specified for non-root nodes')
        if is_root:
            total_iter_count = self.visits

        if self.visits == 0 and not is_root:
            round_result = self.simulate(env, player_in_turn, action_spaces, max_sim_steps)
        else:
            # Select children until a leaf node is reached (leaf is any node without children)
            if len(self.children) < len(action_spaces[player_in_turn-1]):
                next_agent_id = player_in_turn % env.player_count + 1
                self.children.append(Node() if next_agent_id == actor else OpponentNode())
                selected_child_i = len(self.children) - 1
            else:
                # Select the best child
                selected_child_i = self.select_child(total_iter_count)
            
            selected_child = self.children[selected_child_i]
            action = action_spaces[player_in_turn-1][selected_child_i]
            # TODO: What if the game ends here?
            obs, reward, done, info = env.step(env_util.build_action(action, len(action_spaces), player_in_turn))
            if configuration.visualise_all:
                env.render(mode='human', observer='global')
            
            if done:
                winner = env_util.find_winner(info)
                if winner == -1:
                    winner = actor
                if winner == actor:
                    round_result = 1
                else:
                    round_result = 2*max_sim_steps
            # Update the average score and backpropagate the result
            else:
                round_result = selected_child.iterate(env, actor, player_in_turn % env.player_count + 1, action_spaces, total_iter_count, max_sim_steps-1, is_root=False)

        self.visits += 1
        self.avg_score = (self.avg_score * (self.visits - 1) + round_result) / self.visits
        return round_result + 1


    def select_child(self, total_iter_count):
        child_selection_scores = list(map(lambda child: child.calculate_selection_score(total_iter_count), self.children))
        best_selection_score = max(child_selection_scores)
        best_children_idx = [i for i, score in enumerate(child_selection_scores) if abs(score-best_selection_score) < SELECTION_EPSILON]
        return random.choice(best_children_idx)


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
        # Simulate the game
        for _ in range(max_sim_steps):
            # Select a random action
            action = random.choice(action_spaces[current_agent-1])
            obs, reward, done, info = env.step(env_util.build_action(action, len(action_spaces), current_agent))
            if configuration.visualise_all:
                env.render(mode='human', observer='global')
            step_count += 1
            current_agent = current_agent % env.player_count + 1
            if done:
                winner = env_util.find_winner(info)
                # If the environment is done but there is no winner, we assume it was a good result
                if winner == -1:
                    winner = player_in_turn
                if winner == player_in_turn:
                    return step_count
                else:
                    return 2*step_count
        return 2*step_count


    def calculate_selection_score(self, total_iters):
        # Exploration-exploitation trade-off
        exploitation = -self.avg_score
        exploration_raw = sqrt(log(total_iters) / self.visits)
        exploration = EXPLORATION_FACTOR * exploration_raw
        return exploitation + exploration


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
