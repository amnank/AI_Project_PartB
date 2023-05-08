# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir, constants
from .infexion_logic import InfexionGame
from .agent_network import AgentNetwork
from .alpha_zero_helper import create_input, sample_policy

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
            "is_randomized": True,
            "load_network": "Network1",
            "input_depth": 14
        }

        self.board = InfexionGame()
        self.network = AgentNetwork(hyper_params, "Network1")

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """
        inp = create_input(self._color, self.board)
        policy = self.network.get_policy(inp)
        action = sample_policy(policy)

        return action

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """
        