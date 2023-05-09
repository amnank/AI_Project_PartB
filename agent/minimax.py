class Node:
    



class MiniMax:
    def __init__(self, game):
        self.game = game

    def max_value(state, game, alpha, beta):
        if cutoff_test(state):
            return eval(state)
        
        for each child in state.children():

    
