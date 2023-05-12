import numpy as np


class Node:
    """ A node represents a board state in the game """

    def __init__(self, board_state, player):
        self.value = eval(self)
        self.board_state = None
        self.max_player
        self.children = []

        # total powers on the board
        self.player_power = 0
        self.opponent_power = 0

        # 


    def eval(self) -> int:
        """
        The evaluation function. 
        """
        #TODO: write eval function here
        value = 0

        return value
    
    # This is a diluted version of the eval function
    # for best rdering 
    def rough_eval(self) -> int:




    # Added so that the function below shows no errors
    def is_game_ended(self):
        pass


    #TODO: import required game functions and fix accrodingly 
    def generate_successors(self):
        """This function generates the child states of this node and adds them
        to a self.children attribute
        """
    
        state = self.board_state

        # Generate the successors for each cell move in the board
        for cell_position, cell_state in state.items():
            if not cell_state[0] == self.max_player:
                continue

            # Generate a node for each spread direction
            for spread_direction in SPREAD_DIRECTIONS:
                new_board_state = perform_spread(state, cell_position, spread_direction)

                # Create a new node
                new_node = Node(new_board_state)

                self.children.append(new_node)



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

        # this is used to store evaluated state values incase they come up again, avoids recalculation
        self.state_evals = {}

    
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



    
