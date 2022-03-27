import random


class CollectorActionSpaceBuilder:
    available_action_amounts = {
        'idle': 1,
        'lateral_move': 4,
        'diagonal_move': 4,
        'lateral_collect_move': 4,
        'diagonal_collect_move': 4,
        'lateral_push_move': 4,
        'diagonal_push_move': 4,
        'collect': 1,
        'collect_from_ahead': 1,
        'rotate': 4,
        'melee_attack': 1,
        'ranged_attack': 1
    }

    blocked_pairs = {
        'lateral_move': ['lateral_collect_move'],
        'diagonal_move': ['diagonal_collect_move'],
        'lateral_collect_move': ['collect'],
        'diagonal_collect_move': ['collect'],
        'melee_attack': ['ranged_attack'],
    }

    depends_on_either = {
        'rotate': ['melee_attack', 'ranged_attack', 'collect_from_ahead'],
        'collect_from_ahead': ['rotate', 'lateral_move']
    }

    max_number_of_actions = 6

    def build_player_action_space(self):
        valid_action_set = False
        while not valid_action_set:
            action_set_candidate = self.build_random_action_set()
            valid_action_set = self.player_set_validation(action_set_candidate)
        return action_set_candidate


    def build_npc_action_space(self):
        valid_action_set = False
        while not valid_action_set:
            action_set_candidate = self.build_random_action_set()
            valid_action_set = self.npc_set_validation(action_set_candidate)
        return action_set_candidate


    def build_random_action_set(self, probability_coefficient=1.0):
        action_space = set()
        available_actions = self.available_action_amounts.keys()
        uniform_probability = 1.0 / len(available_actions)
        for action_name in available_actions:
            random_float = random.random()
            if random_float < uniform_probability * probability_coefficient:
                action_space.add(action_name)
        return action_space


    def player_set_validation(self, action_set_candidate):
        def contains_collect_actions(action_set):
            collect_actions = {'lateral_collect_move', 'diagonal_collect_move', 'collect', 'collect_from_ahead'}
            found_collect_actions = action_set.intersection(collect_actions) 
            return len(found_collect_actions) > 0

        def contains_move_actions(action_set):
            move_actions = {'lateral_move', 'diagonal_move', 'lateral_collect_move', 'diagonal_collect_move'}
            found_move_actions = action_set.intersection(move_actions)
            return len(found_move_actions) > 0

        return (contains_collect_actions(action_set_candidate) and 
                contains_move_actions(action_set_candidate) and 
                self.total_number_of_actions(action_set_candidate) <= self.max_number_of_actions and
                not self.has_blocked_combos(action_set_candidate) and
                self.satisfies_dependencies(action_set_candidate))


    def npc_set_validation(self, action_set_candidate):
        def contains_turn_and_attack(action_set):
            attack_actions = {'melee_attack', 'ranged_attack'}
            found_collect_actions = action_set.intersection(attack_actions) 
            return len(found_collect_actions) > 0 and 'rotate' in action_set

        def contains_move_actions(action_set):
            move_actions = {'lateral_move', 'diagonal_move', 'lateral_collect_move', 'diagonal_collect_move', 'lateral_push_move', 'diagonal_push_move'}
            found_move_actions = action_set.intersection(move_actions)
            return len(found_move_actions) > 0

        return ((contains_turn_and_attack(action_set_candidate) or contains_move_actions(action_set_candidate)) and 
                self.total_number_of_actions(action_set_candidate) <= self.max_number_of_actions and 
                not self.has_blocked_combos(action_set_candidate) and
                self.satisfies_dependencies(action_set_candidate))


    def total_number_of_actions(self, action_list):
        return sum(self.available_action_amounts[action] for action in action_list)


    def has_blocked_combos(self, action_list):
        for action in action_list:
            if action in self.blocked_pairs:
                for blocked_action in self.blocked_pairs[action]:
                    if blocked_action in action_list:
                        return True
        return False

    def satisfies_dependencies(self, action_list):
        for action in action_list:
            if action in self.depends_on_either:
                satisfied = False
                for dependency in self.depends_on_either[action]:
                    if dependency in action_list:
                        satisfied = True
                        break
                if not satisfied:
                    return False
        return True