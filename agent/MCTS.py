import math

class SearchNode:
    def __init__(self, prior, to_play):
        self.prior = prior          # The probability of taking this action from the parent state
        self.to_play = to_play      # The phasing player

        self.children = {}          # All legal children that can be reached
        self.visit_count = 0        # Times this node was visited during MCTS
        self.total_value = 0        # The overall value of this state
        self.state = None           # The board state at this node

    def value(self):
        return self.total_value / self.visit_count
    

class MCTS:

    def __init__(self, c_puct):
        self.visited = []
        self.Q_values = []
        self.policies = []
        self.num_actions = []

        self.c_puct = c_puct

    def search(self, state, game, neuralnet):
        if game.gameended(state):
            return -game.getreward(state)
    
        if state not in self.visited:
            self.visited.append(state)
            self.policies[state], value = neuralnet.getPredictions(state)
            return -value
        
        max_u, best_action = -float("inf"), -1
        for action in game.getValidActions(state):
            u = self.Q_values[state][action] + self.c_puct * self.policies[state][action]*math.sqrt(sum(self.num_actions[state]))/1+self.num_actions[state][action]
            if u > max_u:
                max_u  = u
                best_action = action

        action = best_action

        next_state = game.nextState(state, action)
        value = self.search(next_state, game, neuralnet)

        self.Q_values[state][action] = (self.num_actions[state][action]*self.Q_values[state][action] + value)/(self.num_actions[state][action]+1)
        self.num_actions[state][action] += 1

        return -value
