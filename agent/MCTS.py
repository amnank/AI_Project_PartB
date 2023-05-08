import math
import numpy as np
import scipy
from agent.neuralnet import AgentNetwork
import torch.optim as optim
from alpha_zero_helper import policy_actions
from agent_game import AgentGame

# hyperparameters
args = {'num_MCTS_simulations': 50,
         'C': 1
        }        


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
    

    def expand(self, policy):
        """
        We expand a node and keep track of the prior policy probability given by the neural network.
        """

        # zip policy and actions
        valid_actions = filter(self.game.is_valid_move(self.player), policy_actions)
        action_probs = zip(valid_actions, policy)
        
        for action, prob in enumerate(action_probs):

            self.game.handle_valid_action(self.player, action)
            next_state = self.game.get_canonical_board()

            # checks valid move 
            if prob != 0:
                self.children[action] = Node(prior=prob, player=self.player * -1, state=next_state) # player switcher


class MCTS:

    def __init__(self, game, network):

        self.thought_board = game       # the Game
        self.network = network          # Neural Network 


    def run(self, network, player, actual_board):
        
        root = Node(self.thought_board, player, prior = 0)

        # iterate through simulations
        for _ in range(self.args['num_MCTS_simulations']):      # This is set to 1600 in the research paper
            node = root
            path = [node]

            # keep selecting next child until we reach an unexpanded node 
            while node.is_fully_expanded():
                action, node = node.select_child()
                path.append(node)
            
            # once we reach a leaf node:
            value = self.game.get_game_ended()

            if value is None:
                # game is not over, get value from network and expand
                value, policy =  network.get_value(), network.get_policy()

                # This makes the prob of invalid moves == 0
                masked_moves = self.thought_board.valid_action_mask(player)
                policy = policy * masked_moves

                node.expand(policy)

            # backup the value
            for node in reversed(path):
                node.value += value     # update reward
                node.visits += 1      

        action_counts = [child.visit_count for child in root.children.values()]
        sum_counts = sum(action_counts)
        policy = [count/sum_counts for count in action_counts]
        
        return policy


class SelfPlay:
    def __init__(self, network, optimizer, game, args): 

        self.network = network
        self.optimizer = optimizer # eg. ADAM

        self.game = game
        self.args = args


        def train_network(self, game):
            examples = []    

            for iteration in range(num_iters):
                for ep in range(num_eps):
                    examples += executeGame(game, nnet)             # collect examples from this game

                new_nnet = network.train_nnet(examples)        

                frac_win = pit(new_nnet, nnet)                      # compare new net with previous net

                if frac_win > threshold: 
                    nnet = new_nnet                                 # replace with new net  

            return nnet
        
        def executeGame(self, nnet):
            """
            """
            game = AgentGame()         # starts a new game
            mcts = MCTS(game, nnet)    

            while True:
                mcts.run(nnet)








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