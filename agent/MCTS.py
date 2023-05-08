import math
import numpy as np
import scipy
from agent.neuralnet import AgentNetwork
import torch.optim as optim

args = {'num_MCTS_simulations': 50,
         'C': 1
        }        # some hyperparameters

class Node:
    """
    This is the Node class (represents a node in the search tree)
    """

    def __init__(self, game, player, prior = 0):

        self.player = player
        self.game = game

        self.visit_count = 0              # number of times thus node was visited in MCTS
        self.value_sum = 0     
        self.prior = prior                # prior probability

        self.children = {}                # All legal children that can be reached    
    

    def is_fully_expanded(self) -> bool:

        # if there are no expandable moves - if node has no children

        return len(self.children) > 0


    def select_child(self):
        """
        Selects child with highest UCB score
        """

        best_score = -np.inf

        for action, child in self.children.items():
            score = self.ucb_score(self, child)

            if score > best_score:
                best_action = action
                best_score = score
                best_child = child
        
        return best_action, best_child
    
       
    def ucb_score(self, child):
        """
        Calculates UCB score
        """
        
        if child.visit_count == 0:
            q_value = 0
        else:
            # to ensure its between 0 and 1
            q_value = 1 - (child.value_sum / child.visit_count) + 1 / 2

        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) /  child.visit_count) * child.prior
    

    def expand(self, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by the neural network.
        """

        for action, prob in enumerate(action_probs):
            self.game.handle_valid_action(self.player, action)
            next_state = self.game.get_canonical_board

            # checks valid move (won't actually ever be zero)
            if prob != 0:
                self.children[action] = Node(prior=prob, player=self.player * -1, state=next_state) # player switcher


class MCTS:

    def __init__(self, game, network):

        self.game = game        # the Game
        self.network = network  # Neural Network 


    def run(self, network, state, player):
        
        root = Node(self.game, player, prior = 0)
        value, policy = network.get_value, network.get_policy
        
        # expand the root to get the children
        root.expand(state, player, policy)

        # iterate through simulations
        for _ in range(self.args['num_MCTS_simulations']):      # This is set to 1600 in the research paper
            node = root
            path = [node]

            # keep selecting next child until we reach an unexpanded node 
            while node.is_fully_expanded():
                action, node = node.select_child()
                path.append(node)
            
            # once we reach a leaf node:
            value = self.game.get_game_ended

            if value is None:
                # game is not over, get value from network and expand
                value, policy =  network.get_value, network.get_policy
                node.expand(policy)

            # backup the value
            for node in path:
                node.value += value
                node.visits += 1


        

            # Players always play from their own perspective
            next_state, _ = self.game.handle_action(state, player, action)
            
            # get board from perspective of other player
            next_state = self.game.get_canonical_board(next_state, player = -1)

            # The value of the new state from the perspective of the other player
            # This value will be None if the game hasnt ended
            value = self.game.get_reward_for_player(next_state, player = 1)

            if value is None:
                # If the node hasnt been explored
                # ask the neural network

                # EXPAND
                policy, value = model.get_policy(next_state), model.get_value(next_state)
                valid_moves = self.game.get_valid_moves(next_state, player)

                # figure out how to mask invalid moves
                node.expand(next_state, parent.player * -1, policy)

            self.backpropagate(search_path, value, parent.player * -1)
        
        return root
    

    def backpropagate(self, search_path, value, player):
        """
        At the end of every simulation of the MCTS, we propagate the evaluation all the way up the tree to the root
        """

        for node in reversed(search_path):
            node.value_sum += value if node.player == player else -value
            node.visit_count += 1


class AlphaZero:
    def __init__(self, model, optimizer, game, args): 
        self.model = model
        # eg. ADAM
        self.optimizer = optimizer

        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

        def train(game, n_epochs):
            for epoch in range(n_epochs):
                # collect training data by self_play
                examples = []
                for 



        def self_play(self):
            training_data = []

            player = []
            state = self.game.get_initial_state()

            while True:
                neutral_state = self.game.player_switcher(state, player)
                action_probs = self.mcts.search(neutral_state)

                training_data.append((neutral_state, action_probs, player))
        
        def executeEpisode(game, nnet):
            training_set = []
            state = self.game.get_initial_state()

            mcts = MCTS(game, args, nnet)

            while True:
                for _ in range(self.args['numMCTS_iterations']):
                    mcts.search(nnet, state, player)
                training_set.append([state, ])
        


        def learn(self):
            for iteration in range(self.args['num_iterations']):
                training_data_for_iteration = []

                for iteration in range(self.args['num_self_play_iterations']):
                    training_data_for_iteration += self.self_play()

                self.model.train()
                for epoch in range(self.args['num_epochs']):
                    self.train(training_data_for_iteration)
                
                # at the end of each iteration, we want to











########### ALTERNATIVE SEARCH FUNCTION (KEEP UNTIL MCTS FINALIZED)




    # def search(self, state):

    #     root = Node(self.game, self.args, state)

    #     for search in range(self.args['num_searches']):
    #         node = root
            
    #         # selection
    #         while node.is_fully_expanded():
    #             node = node.select()
            
    #         value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
    #         value = self.game.get_opponent_value(value)

    #         if not is_terminal:

    #             # policy is logit currently - want to convert it to a distribution of priors
    #             policy = scipy.special.softmax(policy)

    #             valid_moves = self.game.get_valid_moves(node.state)
    #             policy *= valid_moves
    #             policy /= np.sum(policy)

    #             value = value.item()

    #             node = node.expand(policy)   

    #         node.backpropagate(value)
        
    #     action_probs = np.zeros(self.game.action_size)

    #     for child in root.children:
    #         action_probs[child.action_taken] = child.visit_count
    #     action_probs /= np.sum(action_probs)

    #     return action_probs




    
    # def select(self):
    #     """
    #     Selects child with best UCB score
    #     """

    #     best_child = None
    #     best_ucb = -np.inf

    #     for child in self.children:
    #         ucb = self.get_ucb(child)
    #         if ucb > best_ucb:
    #             best_child = child
    #             best_ucb = ucb
        
    #     return best_child 




# class SearchNode:
#     def __init__(self, prior, to_play):
#         self.prior = prior          # The probability of taking this action from the parent state
#         self.to_play = to_play      # The phasing player

#         self.children = {}          # All legal children that can be reached
#         self.visit_count = 0        # Times this node was visited during MCTS
#         self.total_value = 0        # The overall value of this state
#         self.state = None           # The board state at this node

#     def value(self):
#         return self.total_value / self.visit_count