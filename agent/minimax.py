import numpy as np

class Node:
    def __init__(self) -> None:
        pass

# initial values of alpha and beta 
MAX, MIN = np.inf, -np.inf

class MiniMax:

    def __init__(self, game):
        self.game = game

    
    def get_best_val(self, node, depth, isMaxPlaying, alpha, beta):
        
        # If node is a leaf node
        if leaf_node(node):
            return get_value(node)
        
        # MAX is playing
        if isMaxPlaying:
            best_val = MIN
            for child in node.children:
                value = self.get_best_val(child, depth + 1, False, alpha, beta)
                best_val = max(best_val, value)
                alpha = max(alpha, best_val)

                if beta <= alpha:
                    break

            return best_val

        # MIN is playing
        else:
            best_val = MAX
            for node in node.children:
                value = self.get_best_val(child, depth + 1, True, alpha, beta)
                best_val = min(best_val, value)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            return best_val


    
