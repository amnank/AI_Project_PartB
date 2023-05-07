import math
import numpy as np
import scipy

class SearchNode:
    def __init__(self, prior, to_play):
        self.prior = prior          # The probability of taking this action from the parent state
        self.to_play = to_play      # The phasing player

        self.children = {}          # All legal children that can be reached
        self.visit_count = 0        # Times this node was visited during MCTS
        self.total_value = 0        # The overall value of this state
        self.state = None           # The board state at this node

    def value(self):
        return self.total_value / self.visit_count
    

# define a class for a node: We've sorta done this before for Project A (can delete and replace this later)
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior = 0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = 0
        self.value_sum = 0
    

    def is_fully_expanded(self) -> bool:
        # if there are no expandable moves
        # if it has no children
        return len(self.children) > 0
    

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        return best_child 
    
       
    def get_ucb(self, child):
        
        if child.visit_count == 0:
            q_value = 0
        else:
            # to ensure its between 0 and 1
            q_value = 1 - (child.value_sum / child.visit_count) + 1 / 2

        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) /  child.visit_count) * child.prior
    

    def expand(self, policy):
        # this is like spreading or spawning
        for action, prob in enumerate(policy):
            
            child_state = self.state.copy()
            child_state = self.game.get_next_state(child_state, action, 1)
            child_state = self.game.player_switcher(child_state, player = -1)

            child = Node(self.game, self.args, child_state, self, action, prob)
            self.children.append(child)

        return child
    


    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:

    def __init__(self, game, args, model):
        self.args = args
        self.game = game

        # neuralnet outputs
        self.model = model


    def search(self, state):

        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root
            
            # selection
            while node.is_fully_expanded():
                node = node.select()
            
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                #TODO: figure out how to run neural network to get p, v
                policy, value = self.model.get_policy, self.model.get_value 

                # policy is logit currently - want to convert it to a distribution of priors
                policy = scipy.special.softmax(policy)

                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node = node.expand(policy)   

            node.backpropagate(value)
        
        action_probs = np.zeros(self.game.action_size)

        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)

        return action_probs


                # expansion
                # simulation

            # backpropogation


class AlphaZero:
    def __init__(self, model, optimizer, game, args): 
        self.model = model

        # eg. ADAM
        self.optimizer = optimizer

        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

        def self_play(self):
            training_data = []
            player = []
            state = self.game.get_initial_state()

            while True:
                neutral_state = self.game.player_switcher(state, player)
                action_probs = self.mcts.search(neutral_state)

                training_data.append((neutral_state, action_probs, player))
        
        def train(self, training_data):
            pass

        def learn(self):
            for iteration in range(self.args['num_iterations']):
                training_data_for_iteration = []

                for iteration in range(self.args['num_self_play_iterations']):
                    training_data_for_iteration += self.self_play()

                self.model.train()
                for epoch in range(self.args['num_epochs']):
                    self.train(training_data_for_iteration)
                
                # at the end of each iteration, we want to

