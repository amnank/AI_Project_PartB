# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent
from referee.game import \
    PlayerColor, Action
from .agent_network import AgentNetwork
from .alpha_zero_logic import MCTS
from .alpha_zero_helper import greedy_select_from_policy, sample_policy, create_input
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
        match color:
            case PlayerColor.RED:
                # print("Testing: I am playing as red")
                # hyper_params = {
                #     "is_randomized": False,
                #     "load_network": "Network 1 - New Adam",
                #     "input_depth": 14
                # }
                # self.network = AgentNetwork(hyper_params, "GameNet1")
                # self.mcts = MCTS(self.network, 10)
                self.agent = MiniMaxPruning(2)
            case PlayerColor.BLUE:
                # print("Testing: I am playing as blue")
                # hyper_params = {
                #     "is_randomized": False,
                #     "load_network": "Network 1 - Adam",
                #     "input_depth": 14
                # }
                self.agent = MiniMaxPruning(2)


        self.board = GameBoard()
        # self.network = AgentNetwork(hyper_params, "GameNet1")
        # self.mcts = MCTS(self.network, 15)

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """
        # match self._color:
        #     case PlayerColor.RED:
        #         # temp = 0.2
        #         # next_policy = self.mcts.search(temp=temp)
        #         # value = self.network.get_value(create_input(self._color, self.board))
        #         # print(f"Value: {value}")
        #         # if self.board.moves_played <= 30:
        #         #     temp = 1
        #         # elif self.board.moves_played > 30 and self.board.moves_played <= 100:
        #         #     temp = 0.5
        #         # elif self.board.moves_played > 100 and self.board.moves_played <= 200:
        #         #     temp = 0.2
        #         # else:
        #         #     temp = 1e-5

        #         # action = sample_policy(next_policy)
        #         return action
        #     case PlayerColor.BLUE:
        #         return self.agent.run_minimax(self._color, self.board)
        return self.agent.run_minimax(self._color, self.board)

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """
        # if self._color == PlayerColor.RED:
        #     self.mcts.update_tree(action)
        # else:
        self.board.handle_valid_action(color, action)
        