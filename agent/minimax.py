import numpy as np

class Node:
    """ A node represents a board state in the game """

    def __init__(self):
        self.value = eval(self)
        self.board = None
        self.children = []



    def eval(self) -> int:
        """
        The evaluation function. 
        """
        value = 0

        return value

    # Added so that the function below shows no errors
    def is_game_ended(self):
        pass




# TODO: Need to put node.children in the best possible order. 
# tip - moves should be ordered from best to worst for the player whos turn it is 
#       MAX -> order: [highest val ..... lowest val]
#       MIN -> ordeR: [lowest val ...... highest val]
class MiniMaxPruning:

    def __init__(self, game):
        self.game = game

        # TODO: Decide on a depth and set this constant
        self.initial_depth = 0
        self.MAX = np.inf
        self.MIN = -np.inf

    
    def get_best_val(self, node: Node, depth, isMaxPlaying, alpha, beta):
        
        # If node is a leaf node
        if depth == 0 or self.is_game_ended():
            return node.eval()
        
        # MAX is playing
        if isMaxPlaying:
            max_val = self.MIN

            for child in node.children:
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

            for child in node.children:
                eval_value = self.get_best_val(child, depth - 1, True, alpha, beta)
                min_val = min(min_val, eval_value)
                beta = min(beta, eval_value)

                # pruning 
                if beta <= alpha:
                    break
    
            return min_val
        
    
    # Wrapper function
    def minimax(self, node, depth):
        self.get_best_val(node, depth, True, self.MIN, self.MAX)



    
