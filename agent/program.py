# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent
from referee.game import \
    PlayerColor, Action
from .agent_network import AgentNetwork
from .alpha_zero_logic import MCTS
from .alpha_zero_helper import greedy_select_from_policy, sample_policy, create_input
from .infexion_logic import GameBoard

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
                hyper_params = {
                    "is_randomized": False,
                    "load_network": "Network 1",
                    "input_depth": 14
                }
            case PlayerColor.BLUE:
                print("Testing: I am playing as blue")
                hyper_params = {
                    "is_randomized": False,
                    "load_network": "Network 0",
                    "input_depth": 14
                }


        self.board = GameBoard()
        self.network = AgentNetwork(hyper_params, "GameNet1")
        self.mcts = MCTS(self.network, 15)

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """

        next_policy = self.mcts.search()
        value = self.network.get_value(create_input(self._color, self.board))
        print(f"Value: {value}")
        if self.board.moves_played <= 30:
            action = sample_policy(next_policy)
        else:
            action = greedy_select_from_policy(next_policy)
        return action

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """
        self.mcts.update_tree(action)
        self.board.handle_valid_action(color, action)
        