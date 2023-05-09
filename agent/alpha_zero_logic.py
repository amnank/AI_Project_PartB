import math
import numpy as np
from agent_network import AgentNetwork        # pylint: disable=import-error
from alpha_zero_helper import\
    policy_actions , get_symmetries, create_input, sample_policy, greedy_select_from_policy   # pylint: disable=import-error
from infexion_logic import InfexionGame, GameBoard             # pylint: disable=import-error
from referee.game import \
    PlayerColor, SpawnAction, SpreadAction


# hyperparameters
mcts_args = {'num_MCTS_simulations': 50,
         'C': 1
        }

infexion = InfexionGame()

class Node:
    """
    This is the Node class (represents a node in the search tree)
    """

    def __init__(self, player:'PlayerColor', game_board:'GameBoard', path, prior = 0):

        self.player = player
        self.game_board = game_board

        # number of times this node was visited in MCTS
        self.visit_count = 0

        # prior probability
        self.value_sum = 0
        self.prior = prior

        # All legal children that can be reached
        self.children = {}

        # Path to root (action, start_state)
        self.path = path

    def is_fully_expanded(self) -> bool:
        """This function checks if a node is fully expanded or not
        """
        # if there are no expandable moves - if node has no children
        return len(self.children) > 0

    def select_child(self) -> tuple('SpawnAction|SpreadAction', 'GameBoard'):
        """
        Selects child with highest UCB score and returns the key, value pair
        """
        best_score = -np.inf

        for action, child in self.children.items():
            score = self._ucb_score(child)

            if score > best_score:
                best_action = action
                best_score = score
                best_child = child
        
        return (best_action, best_child)
    
    def _ucb_score(self, child):
        """
        Calculates UCB score
        """
        
        if child.visit_count == 0:
            q_value = 0
        else:
            # to ensure its between 0 and 1
            q_value = 1 - (child.value_sum / child.visit_count) + 1 / 2

        sum_counts = sum([child.visit_count for child in self.children.values()])
        return q_value + mcts_args['C'] * math.sqrt(sum_counts / (1 + child.visit_count)) * child.prior
    

    def expand(self, policy):
        """
        Generates all successor states of this Node and adds them as its children
        """

        # zip policy and actions
        action_probs = zip(policy_actions, infexion.valid_action_mask(self.game_board, self.player), policy)
        
        for action, is_valid, prob in enumerate(action_probs):
            if is_valid:
                # Execute the action on a new game board
                next_game_board = GameBoard(self.game_board)
                next_game_board.handle_valid_action(self.player, action)

                # Switch players and create a new node
                self.children[action] = Node(self.player.opponent, next_game_board, self.path + [(action, self)], prob)
            else:
                self.children[action] = Node(self.player.opponent, self.game_board, self.path + [(action, self)])


class MCTS:
    """This class encapsulates the functionality of the Monte Carlo Tree Search
    """
    def __init__(self, self_play_mode:bool):
        # Initialize the root node with the player as RED, starting board and prior of 0
        self.root = Node(PlayerColor.RED, GameBoard(), [], 0)
        self.self_play_mode = self_play_mode


    def run(self, network:'AgentNetwork'):
        """This funcion runs the specified number of iterations of MCTS and
            picks an action to play

        Args:
            current_player (PlayerColor): The phasing player
            actual_board (GameBoard): The board state to output the policy from

        Returns:
            343 x 1: Policy vector
        """

        # iterate through simulations
        for _ in range(mcts_args['num_MCTS_simulations']):
            curr_node = self.root

            # keep selecting next child until we reach an unexpanded node
            while curr_node.is_fully_expanded():
                _, child_node = curr_node.select_child()

           # once we reach a leaf node:
            value = infexion.get_game_ended(child_node.game_board)

            if value is None:
                # game is not over, get value from network and expand
                inp = create_input(child_node.player, child_node.game_board)
                value, policy = network.get_value(inp), network.get_policy(inp)
                child_node.expand(policy)

            # backup the value
            for _, back_node in reversed(child_node.path):
                # update reward
                if back_node.player == child_node.player:
                    back_node.value += value
                else:
                    back_node.value += -value

                back_node.visits += 1

        # Calculate the improved policy
        action_counts = [child.visit_count for child in self.root.children.values()]
        sum_counts = sum(action_counts)
        improved_policy = [count/sum_counts for count in action_counts]

        if self.self_play_mode:
            chosen_action = sample_policy(improved_policy)
        else:
            chosen_action = greedy_select_from_policy(improved_policy)

        self.root = self.root.children[chosen_action]

        return chosen_action
    
    def update_state(self, last_action_taken:'SpreadAction|SpawnAction|None'):
        """Call this function when the opponent performs an action that you have
        to update in your own representation of the board

        Args:
            last_action_taken (SpreadAction|SpawnAction|None, optional): Last Action Taken.
        """
        if last_action_taken is not None:
            self.root = self.root.children[last_action_taken]

self_play_args = {
    'num_iters': 80,
    'num_train_games': 100,
    'pit_games': 40,
    'threshold': 0.6
}

class SelfPlay:
    """This class contains the functionality required for the Algorithm to play
    against itself and train itself
    """
    def __init__(self, optimizer):
        
        self.hyper_params={
            "is_randomized": True,
            "load_network": None,
            "input_depth": 16 
        }
        
        self.network = AgentNetwork(self.hyper_params, "Network 0")
        self.optimizer = optimizer # eg. ADAM

    def train_network(self, should_dump:bool):
        """This function trains the neural network using the AlphaGo Zero 
        Approach and returns the final trained network, while dumping all
        previous ones
        """
        for i in range(self_play_args['num_iters']):
            self.network.network_name = f"Network {i}"
            if should_dump:
                self.network.save_network()
            new_nnet = self._execute_iteration(self.network)
            self.network = new_nnet
        
        self.network.network_name = f"Network {i}"
        if should_dump:
                self.network.save_network()
        return self.network

    def _execute_iteration(self, nnet):
        examples = []
        frac_win = 0

        while frac_win < self_play_args['threshold']:
            for _ in range(self_play_args['num_train_games']):
                # collect examples from this game
                examples += self._execute_game(nnet)

            new_nnet = nnet.train_nnet(examples)

            frac_win = self._pit(new_nnet, nnet)

        return new_nnet
        
    def _execute_game(self, nnet:AgentNetwork):
        """This function plays out one full game of self play and generates
        training examples from the moves played

        Args:
            nnet (AgentNetwork): The neural net playing against itself

        Returns:
            list(tuple): (state, action_played, resultant value)
        """
        examples = []
        # starts a new game
        mcts = MCTS(self_play_mode=True)
        game_board = GameBoard()
        curr_player = PlayerColor.RED

        while True:
            next_action = mcts.run(nnet)
            examples.append([create_input(curr_player, game_board), next_action, curr_player])
            game_board.handle_valid_action(curr_player, next_action)
            curr_player = curr_player.opponent
            
            val = infexion.get_game_ended(game_board)
            if val is not None:
                break
        
        for example in examples:
            if val == 0:
                example[2] = 0
            else:
                example[2] = 1 if int(example[2]) == val else -1
        

        sym_examples = []
        # Implement symmetries
        for example in examples:
            sym_list = [get_symmetries(inp) for inp in example[0]]
            sym_list = np.transpose(sym_list, (1, 0 , 2, 3))
            for variation in sym_list:
                sym_examples.append([variation, example[1], example[2]])



        examples_tuples = [tuple(example) for example in examples]


        return examples_tuples
    
    def _pit(self, new_nnet:'AgentNetwork', old_nnet:'AgentNetwork'):

        new_nnet_won = 0
        old_nnet_won = 0
        total_games = self_play_args['pit_games']

        for _ in range(total_games):
            mcts_new = MCTS(self_play_mode=False)
            mcts_old = MCTS(self_play_mode=False)

            game_board = GameBoard()
            curr_player = PlayerColor.RED
            winner = None

            curr_bot = np.random.choice([(new_nnet, mcts_new), (old_nnet, mcts_old)])
            curr_mcts, curr_nnet = curr_bot

            red_player = curr_bot
            blue_player = (new_nnet, mcts_new) if curr_bot == (old_nnet, mcts_old) else (old_nnet, mcts_old)

            while True:
                next_action = curr_mcts.run(curr_nnet)
                game_board.handle_valid_action(curr_player, next_action)
                curr_player = curr_player.opponent

                curr_bot = (new_nnet, mcts_new) if curr_bot == (old_nnet, mcts_old) else (old_nnet, mcts_old)
                curr_mcts, curr_nnet = curr_bot
                curr_mcts.update_state(next_action)
                
                val = infexion.get_game_ended(game_board)
                if val is not None:
                    winner = val
                    break
            
            if winner != 0:
                if winner == int(PlayerColor.RED):
                    new_nnet_won += 1 if red_player[0] == new_nnet else 0
                    old_nnet_won += 1 if red_player[0] == old_nnet else 0
                else:
                    new_nnet_won += 1 if blue_player[0] == new_nnet else 0
                    old_nnet_won += 1 if blue_player[0] == old_nnet else 0


        return new_nnet_won / total_games