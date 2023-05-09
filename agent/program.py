# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from referee.game import \
    PlayerColor, Action
from .agent_network import AgentNetwork
from .alpha_zero_logic import MCTS

# This is the entry point for your game playing agent. Currently the agent
# simply spawns a token at the centre of the board if playing as RED, and
# spreads a token at the centre of the board if playing as BLUE. This is
# intended to serve as an example of how to use the referee API -- obviously
# this is not a valid strategy for actually playing the game!

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initialise the agent.
        """
        self._color = color
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as red")
            case PlayerColor.BLUE:
                print("Testing: I am playing as blue")

        hyper_params = {
            "is_randomized": False,
            "load_network": "insert_best_network_here",
            "input_depth": 16
        }

        self.network = AgentNetwork(hyper_params, "GameNet1")
        self.mcts = MCTS(self_play_mode=False)

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """

        action = self.mcts.run(self.network)
        return action

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """
        if color == self._color.opponent:
            self.mcts.update_state(action)
        