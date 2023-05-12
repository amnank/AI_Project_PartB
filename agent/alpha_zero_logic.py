import sys
sys.path.append("game")
import math
import random
import numpy as np
from referee.game import PlayerColor, SpawnAction, SpreadAction # pylint: disable=import-error
from .agent_network import AgentNetwork        # pylint: disable=import-error
from .alpha_zero_helper import\
    policy_actions, valid_action_mask, create_input, sample_policy, greedy_select_from_policy, get_symmetries, get_policy_symmetries   # pylint: disable=import-error
from .infexion_logic import infexion_game, GameBoard             # pylint: disable=import-error
import time 


self_play_args = {
    'num_iters': 10,
    'num_train_games': 10,
    'pit_games': 10,
    'threshold': 0.6
}

class Node:
    """
    This is the Node class (represents a node in the search tree)
    """

    def __init__(self, player:'PlayerColor', game_board:'GameBoard',prior=0, action_index=None, parent=None):
        self.player_to_move = player
        self.game_board = game_board

        # number of times this node was visited in MCTS
        self.visit_count = 0

        # prior probability
        self.prior = prior
        self.value_sum = 0
        self.action_index = action_index

        # All legal children that can be reached
        self.children = []
        self.expandable_moves = np.array([valid_action_mask(self.game_board, self.player_to_move)]).T

        # Path to root (action, start_state)
        self.parent = parent

    def is_fully_expanded(self) -> bool:
        """This function checks if a node is fully expanded or not
        """
        # if there are no expandable moves - if node has no children
        return (len(self.children) > 0 and self.expandable_moves.sum() == 0)

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

    def get_value(self):
        if self.visit_count == 0:
            return 0
        
        return self.value_sum / self.visit_count
    
    def _ucb_score(self, child:'Node'):
        """
        Calculates UCB score
        """
        prior_score  = child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)
        if child.visit_count > 0:
            value_score = -1 * child.get_value()
        else:
            value_score = 0

        return value_score + prior_score
    

    def expand(self, priors):
        """
        Generate a successor state of this Node and add it to its children
        """
        action_idx = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action_idx] = 0
        action = policy_actions[action_idx]

        pol = priors[action_idx][0]

        board = GameBoard(self.game_board)    
        player = self.player_to_move.opponent

        board.handle_valid_action(player, action)
        self.children.append(Node(player, board, pol, action_idx, self))   
    
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = value * -1
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    """This class encapsulates the functionality of the Monte Carlo Tree Search
    """
    def __init__(self, network:'AgentNetwork', sims=20):
        self.network = network
        self.sims = sims

        self.root = Node(PlayerColor.RED, GameBoard())
        self._expand_root()

    def search(self):
        for _ in range(self.sims):
            node = self.root

            while node.is_fully_expanded():
                node = node.select_child()

            value = infexion_game.get_game_ended(node.game_board)
            if value == int(node.player_to_move):
                value = 1
            elif value == int(node.player_to_move.opponent):
                value = -1

            if value is None:
                state = create_input(node.player_to_move, node.game_board)
                value = self.network.get_value(state)
                policy = self.network.get_policy(state)

                policy = np.multiply(policy, node.expandable_moves)
                node.expand(policy)

            node.backpropagate(value)

        action_space_shape = (len(policy_actions), 1)
        visit_counts = np.zeros(action_space_shape)
        action_probs = np.zeros(action_space_shape)

        for child in self.root.children:
            visit_counts[child.action_index] = child.visit_count
            # if child.value_sum < 0:
            #     action_probs[child.action_index] = 0.00000001
            # else:
            #     action_probs[child.action_index] = child.value_sum

        # action_probs /= np.sum(visit_counts)
        action_probs = visit_counts / np.sum(visit_counts)
        return action_probs

    def update_tree(self, last_action:'SpreadAction|SpawnAction'):
        action_idx = np.where(policy_actions == last_action)[0][0]

        for child in self.root.children:
            if child.action_index == action_idx:
                self.root = child
                self.root.parent = None
                self.root.expandable_moves = np.array([valid_action_mask(self.root.game_board, self.root.player_to_move)]).T
                self._expand_root()
                return
        raise ValueError("Root does not have child trying to be updated")

    def _expand_root(self):
        state = create_input(self.root.player_to_move, self.root.game_board)
        priors = self.network.get_policy(state)
        self.root.children = []

        priors = np.multiply(priors, self.root.expandable_moves)

        i = -1
        for pol in np.nditer(priors, order='F'):
            i += 1
            pol = pol.item()
            if pol == 0:
                continue
            
            action = policy_actions[i]
            board = GameBoard(self.root.game_board)

            board.handle_valid_action(self.root.player_to_move, action)
            new_player = self.root.player_to_move.opponent
            self.root.children.append(Node(new_player, board, pol, i, self.root))

        self.root.expandable_moves = np.zeros(self.root.expandable_moves.shape)

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

            print("Starting training")
            old_name = nnet.network_name
            new_nnet = nnet.train(examples)

            hyper_params={
                "is_randomized": False,
                "load_network": old_name,
                "input_depth": 14
            }
            
            print("Starting head to head")
            frac_win = self._pit(new_nnet, AgentNetwork(hyper_params, "Old"))
            print(f"Frac won {frac_win}")


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
            improved_policy = mcts.search()
            next_action = sample_policy(improved_policy)
            # action_index = np.where(policy_actions == next_action)[0][0]
            # print(infexion_game.is_valid_move(game_board, curr_player, next_action), next_action, curr_player)
            mcts.update_tree(next_action)

            if infexion_game.is_valid_move(game_board, curr_player, next_action) is False:
                for r in game_board.get_canonical_board(PlayerColor.BLUE):
                    print(r)
            
            examples.append([create_input(curr_player, game_board), improved_policy, curr_player])
            game_board.handle_valid_action(curr_player, next_action)
            if game_board.moves_played % 50 == 0:
                print(f"{game_board.moves_played} moves played")

            curr_player = curr_player.opponent
            
            val = infexion_game.get_game_ended(game_board)
            if val is not None:
                break

        for example in examples:
            if val == 0:
                example[2] = 0
            else:
                example[2] = 1 if int(example[2]) == val else -1

        examples_tuples = [tuple(example) for example in examples]


        # sym_examples = []
        # Implement symmetries
        # for example in examples:
        #     inp_sym_list = np.array([get_symmetries(board) for board in example[0]])
        #     inp_sym_list = np.transpose(inp_sym_list, (1, 0, 2, 3))
        #     policy_sym_list = get_policy_symmetries(example[1])

        #     for inp_var, policy_var in zip(inp_sym_list, policy_sym_list):
        #         sym_examples.append((inp_var, policy_var, example[2]))

        # examples_tuples += sym_examples
        print(f"Moves played: {game_board.moves_played}")
        return examples_tuples
    
    def _pit(self, new_nnet:'AgentNetwork', old_nnet:'AgentNetwork'):

        new_nnet_won = 0
        old_nnet_won = 0
        draw_count = 0
        total_games = self_play_args['pit_games']
        frac_win = 0

        for i in range(total_games):
            print(f"Head to head, Game {i}")

            game_board = GameBoard()
            curr_player = PlayerColor.RED
            winner = None

            new_mcts = MCTS(new_nnet, 5)
            old_mcts = MCTS(old_nnet, 5)

            red_player = random.choice([new_mcts, old_mcts])
            blue_player = new_mcts if red_player == old_mcts else old_mcts
            curr_mcts = red_player

            while True:
                # Current player plays an action
                next_policy = curr_mcts.search()
                if game_board.moves_played <= 30:
                    next_action = sample_policy(next_policy)
                else:
                    next_action = greedy_select_from_policy(next_policy)

                # print(infexion_game.is_valid_move(game_board, curr_player, next_action), next_action, curr_player)
                # if infexion_game.is_valid_move(game_board, curr_player, next_action) is False:
                #     for r in game_board.get_canonical_board(PlayerColor.BLUE):
                #         print(r)

                curr_mcts.update_tree(next_action)

                # Board and current player's MCTS is updated
                game_board.handle_valid_action(curr_player, next_action)

                if game_board.moves_played % 50 == 0:
                    print(f"{game_board.moves_played} moves played")

                curr_player = curr_player.opponent
                curr_mcts = new_mcts if curr_mcts == old_mcts else old_mcts
                curr_mcts.update_tree(next_action)
                
                val = infexion_game.get_game_ended(game_board)
                if val is not None:
                    old_player = PlayerColor.RED if red_player == old_mcts else PlayerColor.BLUE
                    new_player = PlayerColor.RED if red_player == new_mcts else PlayerColor.BLUE
                    print(f"OLD POWER {str(old_player)}: {game_board.count_power(old_player)}")
                    print(f"NEW POWER {str(new_player)}: {game_board.count_power(new_player)}")
                    winner = val
                    print(f"Winner: {str(winner)}")
                    break
            
            if winner != 0:
                if winner == int(PlayerColor.RED):
                    new_nnet_won += 1 if red_player == new_mcts else 0
                    old_nnet_won += 1 if red_player == old_mcts else 0
                else:
                    new_nnet_won += 1 if blue_player == new_mcts else 0
                    old_nnet_won += 1 if blue_player == old_mcts else 0
                frac_win = new_nnet_won / total_games
            
            if (winner == 0):
                draw_count += 1
            print(f"Old won: {old_nnet_won}/{total_games}")
            print(f"New won: {new_nnet_won}/{total_games}")
            print(f"Drawn: {draw_count}/{total_games}")
            if frac_win > self_play_args["threshold"]:
                return frac_win
            if ((old_nnet_won + draw_count) / total_games) > (1 - self_play_args["threshold"]):
                return frac_win

        return new_nnet_won / total_games