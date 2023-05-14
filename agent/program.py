# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent
from referee.game import \
    PlayerColor, Action
from .infexion_logic import GameBoard
from .minimax import MiniMaxPruning

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
        self.agent = MiniMaxPruning(3, 12000)
        self.board = GameBoard()

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """
        return self.agent.run_minimax(self._color, self.board)

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """
        self.board.handle_valid_action(color, action)
        