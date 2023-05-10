import sys
sys.path.append("game")
import math
import random
import numpy as np
from agent_network import AgentNetwork        # pylint: disable=import-error
from alpha_zero_helper import\
    policy_actions, valid_action_mask, get_symmetries, create_input, sample_policy, greedy_select_from_policy, get_policy_symmetries   # pylint: disable=import-error
from infexion_logic import infexion_game, GameBoard             # pylint: disable=import-error
from game import PlayerColor, SpawnAction, SpreadAction # pylint: disable=import-error
import time 


# hyperparameters
mcts_args = {'num_MCTS_simulations': 25,
         'C': 1
        }

class Node:
    """
    This is the Node class (represents a node in the search tree)
    """

    def __init__(self, player:'PlayerColor', game_board:'GameBoard', action_index=None, parent=None):
        self.player_to_move = player
        self.game_board = game_board

        # number of times this node was visited in MCTS
        self.visit_count = 0
        # prior probability
        self.value_sum = 0
        self.action_index = action_index

        # All legal children that can be reached
        self.children = []
        self.expandable_moves = np.array(valid_action_mask(self.game_board, self.player_to_move))

        # Path to root (action, start_state)
        self.parent = parent

    def is_fully_expanded(self) -> bool:
        """This function checks if a node is fully expanded or not
        """
        # if there are no expandable moves - if node has no children
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select_child(self):
        """
        Selects child with highest UCB score and returns the key, value pair
        """
        best_score = -np.inf
        best_child = None

        for child in self.children:
            score = self._ucb_score(child)

            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _ucb_score(self, child:'Node'):
        """
        Calculates UCB score
        """

        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + mcts_args["C"] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    

    def expand(self):
        """
        Generate a successor state of this Node and add it to its children
        """

        # self.expandable_moves = np.array(valid_action_mask(self.game_board, self.player_to_move))
        try:
            action_index = np.random.choice(np.where(self.expandable_moves == 1)[0])
        except:
            print(self.expandable_moves)
            print(valid_action_mask(self.game_board, self.player_to_move))
            print(self.game_board.get_canonical_board(self.player_to_move))
            print(infexion_game.get_game_ended(self.game_board))
            exit()

        action = policy_actions[action_index]

        self.expandable_moves[action_index] = 0

        child_board = GameBoard(self.game_board)
        child_board.handle_valid_action(self.player_to_move, action)
        child_player = self.player_to_move.opponent

        child = Node(child_player, child_board, action_index, self)
        self.children.append(child)
        return child
    
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = value * -1
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    """This class encapsulates the functionality of the Monte Carlo Tree Search
    """
    def __init__(self, network:'AgentNetwork'):
        self.network = network

    def search(self, player:'PlayerColor', game_board:'GameBoard'):
        root = Node(player, game_board)

        for _ in range(mcts_args["num_MCTS_simulations"]):
            node = root

            while node.is_fully_expanded():
                node = node.select_child()

            value = infexion_game.get_game_ended(node.game_board)
            if value == int(node.player_to_move):
                value = 1
            elif value == int(node.player_to_move.opponent):
                value = -1

            if value is None:
                node = node.expand()
                value = self.network.get_value(create_input(node.player_to_move, node.game_board))

            node.backpropagate(value)

        action_space_shape = (len(policy_actions), 1)
        action_probs = np.zeros(action_space_shape)
        for child in root.children:
            action_probs[child.action_index] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


self_play_args = {
    'num_iters': 5,
    'num_train_games': 40,
    'pit_games': 10,
    'threshold': 0.55
}

class SelfPlay:
    """This class contains the functionality required for the Algorithm to play
    against itself and train itself
    """
    def __init__(self):
        
        self.hyper_params={
            "is_randomized": True,
            "load_network": None,
            "input_depth": 14
        }
        
        print("Created network!")
        self.network = AgentNetwork(self.hyper_params, "Network 0")

    def train_network(self, should_dump:bool):
        """This function trains the neural network using the AlphaGo Zero 
        Approach and returns the final trained network, while dumping all
        previous ones
        """
        self.starttime = time.process_time()
        
        for i in range(self_play_args['num_iters']):
            self.network.network_name = f"Network {i}"
            if should_dump:
                self.network.save_network()
            print(f"Starting Iteration {i}")
            new_nnet = self._execute_iteration(self.network)
            self.network = new_nnet
        
        self.network.network_name = f"Network {i}"
        if should_dump:
                self.network.save_network()
        return self.network

    def _execute_iteration(self, nnet:'AgentNetwork'):
        examples = []
        frac_win = 0

        while frac_win < self_play_args['threshold']:
            for i in range(self_play_args['num_train_games']):
                print(f"Playing game {i}")
                # collect examples from this game
                examples += self._execute_game(nnet)
                ## one game ended
                exittime = time.process_time()
                elapsed = exittime - self.starttime
                print("TIME:", elapsed)
                exit()

            print("Starting training")
            old_name = nnet.network_name
            new_nnet = nnet.train(examples)

            hyper_params={
                "is_randomized": False,
                "load_network": old_name,
                "input_depth": 14
            }
            
            print("Starting head to head")
            new_nnet_won, old_nnet_won, total_games = self._pit(new_nnet, AgentNetwork(hyper_params, "Old"))
            frac_win = new_nnet_won / total_games
            print(f"Frac won {frac_win}")
            if old_nnet_won + new_nnet_won == 0:
                break


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
        mcts = MCTS(nnet)
        game_board = GameBoard()
        curr_player = PlayerColor.RED

        while True:
            improved_policy = mcts.search(curr_player, game_board)
            next_action = sample_policy(improved_policy)

            examples.append([create_input(curr_player, game_board), improved_policy, curr_player])
            game_board.handle_valid_action(curr_player, next_action)
            # if game_board.moves_played % 15 == 0:
            #     print(f"{game_board.moves_played} moves played")

            curr_player = curr_player.opponent
            
            val = infexion_game.get_game_ended(game_board)
            if val is not None:
                board = game_board.get_canonical_board(curr_player)
                for r in board:
                    print(r)
                print()
                print(f"Total power {game_board.count_total_power()}")
                break
        
        for example in examples:
            if val == 0:
                example[2] = 0
            else:
                example[2] = 1 if int(example[2]) == val else -1
        

        # sym_examples = []
        # # Implement symmetries
        # for example in examples:
        #     inp_sym_list = [get_symmetries(inp) for inp in example[0]]
        #     inp_sym_list = np.transpose(inp_sym_list, (1, 0, 2, 3))
        #     policy_sym_list = get_policy_symmetries(example[1])

        #     for inp_var, policy_var in zip(inp_sym_list, policy_sym_list):
        #         # Add policy symmetries
        #         sym_examples.append([inp_var, policy_var, example[2]])
        
        # examples += sym_examples

        examples_tuples = [tuple(example) for example in examples]


        return examples_tuples
    
    def _pit(self, new_nnet:'AgentNetwork', old_nnet:'AgentNetwork'):

        new_nnet_won = 0
        old_nnet_won = 0
        total_games = self_play_args['pit_games']

        for i in range(total_games):
            print(f"Head to head, Game {i}")

            game_board = GameBoard()
            curr_player = PlayerColor.RED
            winner = None

            new_mcts = MCTS(new_nnet)
            old_mcts = MCTS(old_nnet)

            curr_mcts = random.choice([new_mcts, old_mcts])

            red_player = curr_mcts
            blue_player = new_mcts if curr_mcts == old_mcts else old_mcts

            while True:
                # Current player plays an action
                next_policy = curr_mcts.search(curr_player, game_board)
                next_action = greedy_select_from_policy(next_policy)

                # Board and current player's MCTS is updated
                game_board.handle_valid_action(curr_player, next_action)

                if game_board.moves_played % 50 == 0:
                    print(f"{game_board.moves_played} moves played")

                curr_player = curr_player.opponent
                curr_mcts = new_mcts if curr_mcts == old_mcts else old_mcts
                
                val = infexion_game.get_game_ended(game_board)
                if val is not None:
                    winner = val
                    board = game_board.get_canonical_board(curr_player)
                    for r in board:
                        print(r)
                    print()
                    break
            
            if winner != 0:
                if winner == int(PlayerColor.RED):
                    new_nnet_won += 1 if red_player == new_nnet else 0
                    old_nnet_won += 1 if red_player == old_nnet else 0
                else:
                    new_nnet_won += 1 if blue_player == new_nnet else 0
                    old_nnet_won += 1 if blue_player == old_nnet else 0

        return (new_nnet_won, old_nnet_won, total_games)