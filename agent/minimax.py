import numpy as np

class Node:
    def __init__(self) -> None:
        pass

def is_game_ended(node):
    pass




# TODO: Need to put the nodes children in the best possible order.
class MiniMaxPruning:

    def __init__(self, game):
        self.game = game
        
        # TODO: Set this constant
        self.initial_depth = 0
        self.MAX = np.inf
        self.MIN = -np.inf

    def eval(node) -> int:
        """
        The evaluation function. 
        """
        pass

    
    def get_best_val(self, node, depth, isMaxPlaying, alpha, beta):
        
        # If node is a leaf node
        if depth == 0 or is_game_ended(node):
            return eval(node)
        
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

            for node in node.children:
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


    
