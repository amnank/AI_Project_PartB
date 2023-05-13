import numpy as np
from infexion_logic import InfexionGame
from referee.game import PlayerColor, Board

game = InfexionGame()

class Node:
    """ A node represents a board state in the game """

    def __init__(self, board: Board, player: PlayerColor):
        self.value = eval(self)
        self.board = board
        self.max_player
        self.children = game.get_valid_moves(board, player)

        # total powers on the board
        self.player_power = 0
        self.opponent_power = 0


    def eval(self) -> int:
        """
        The evaluation function. 
        """
        #TODO: write eval function here
        value = self.player_power - self.opponent_power

        return value
    

    def order_children(self, isMaxPlaying):
        if isMaxPlaying:
            return self.children
        else:
            return reversed(self.children)


    # Added so that the function below shows no errors
    def is_game_ended(self):
        pass


# tip - moves should be ordered from best to worst for the player whos turn it is 
#       MAX -> order: [highest val ..... lowest val]
#       MIN -> ordeR: [lowest val ...... highest val]

class MiniMaxPruning:

    def __init__(self, player):
        self.game = game
        self.player = player
        self.root = Node(InfexionGame(), Board(), player)

        # TODO: Decide on a depth and set this constant
        self.initial_depth = 0
        self.MAX = np.inf
        self.MIN = -np.inf

        # this is used to store evaluated state values incase they come up again, avoids recalculation
        self.state_evals = {}

    
    def get_best_val(self, node: Node, depth, isMaxPlaying, alpha, beta):
        
        # If node is a leaf node
        if depth == 0 or self.is_game_ended():
            return node.eval()
        
        children = node.order_children(isMaxPlaying)
        # MAX is playing
        if isMaxPlaying:
            max_val = self.MIN

            for child in children:
                eval_value = self.get_best_val(child, depth - 1, False, alpha, beta)
                max_val = max(max_val, eval_value)
                alpha = max(alpha, eval_value)

                # pruning
                if beta <= alpha:
                    break

            return max_val

        # MIN is playing
        else:
            min_val = self.MAX

            for child in children:
                eval_value = self.get_best_val(child, depth - 1, True, alpha, beta)
                min_val = min(min_val, eval_value)
                beta = min(beta, eval_value)

                # pruning 
                if beta <= alpha:
                    break
    
            return min_val
        
    
    # Wrapper function
    def minimax(self):
        self.get_best_val(node, depth, True, self.MIN, self.MAX)

    
