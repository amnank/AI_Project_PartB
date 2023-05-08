import math
import numpy as np
import scipy
from agent.neuralnet import AgentNetwork
import torch.optim as optim


class Node:
    """
    This is the Node class (represents a node in the search tree)
    """

    def __init__(self, game, args, state, parent=None, action_taken=None, prior = 0, player):

        self.player = player
        self.game = game
        self.args = args                  # some hyperparameters
        self.state = state                # the board state at this Node
        self.parent = parent              # HMMM
        self.action_taken = action_taken
        

        self.children = {}                # All legal children that can be reached

        self.visit_count = 0              # N - Number of times thus node was visited in MCTS
        self.value_sum = 0     
        self.prior = prior           
    
    
    def value(self):
        """
        Average value for a node
        """
    
        if self.visit_count == 0:
            value = 0
        else:
            value = self.value_sum / self.visit_count
        
        return value
    

    def is_fully_expanded(self) -> bool:

        # if there are no expandable moves - if node has no children

        return len(self.children) > 0


    def select_child(self):
        """
        Selects child with highest UCB score
        """

        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = self.ucb_score(self, child)

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
       
    def get_ucb(self, child):
        """
        Calculates UCB score
        """
        
        if child.visit_count == 0:
            q_value = 0
        else:
            # to ensure its between 0 and 1
            q_value = 1 - (child.value_sum / child.visit_count) + 1 / 2

        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) /  child.visit_count) * child.prior
    

    def expand(self, state, player, policy):
        """
        We expand a node and keep track of the prior policy probability given by the neural network.
        """
        self.player = player
        self.state = state

        for action, prob in enumerate(policy):
            # checks valid move (won't actually ever be zero)
            if prob != 0:
                self.children[action] = Node(prior=prob, player=self.player * -1) # player switcher


    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:

    def __init__(self, game, args, network):

        self.args = args    # hyperparameters (will fix later)
        self.game = game    # the Infexion game
        self.network = network  # neuralnet 


    def search(self, model, state, player):
        
        root = Node(self.game, self.args, state)

        # EXPAND:

        policy, value = model.get_policy(state), model.get_value(state)
        valid_moves = self.game.get_valid_moves(state, player)
        
        # figure out how to mask invalid moves
        # will have to alter policy vector 
        
        root.expand(state, player, policy)

        for _ in range(self.args['num_simulations']):      # This is set to 1600 in the research paper
            node = root
            search_path = [node]

            # SELECT
            while node.is_fully_expanded():
                action, node = node.select_child()
                search_path.append(node)
            
            parent = search_path[-2]
            state = parent.state

            # Now we're at the leaf node and we would like to expand

            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, player, action)
            
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

        def self_play(self):
            model = AgentNetwork()
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