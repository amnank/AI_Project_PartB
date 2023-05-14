import numpy as np
from .infexion_logic import InfexionGame, GameBoard # pylint: disable=import-error
from referee.game import PlayerColor, constants, SpawnAction, SpreadAction, HexPos, HexDir

game = InfexionGame()

# actions_list = []
# for q in range(constants.BOARD_N):
#     for r in range(constants.BOARD_N):
#         actions_list.append(SpawnAction(HexPos(r,q)))

# for q in range(constants.BOARD_N):
#     for r in range(constants.BOARD_N):
#         actions_list.append(SpreadAction(HexPos(r,q), HexDir.DownLeft))
#         actions_list.append(SpreadAction(HexPos(r,q), HexDir.Down))
#         actions_list.append(SpreadAction(HexPos(r,q), HexDir.DownRight))
#         actions_list.append(SpreadAction(HexPos(r,q), HexDir.Up))
#         actions_list.append(SpreadAction(HexPos(r,q), HexDir.UpLeft))
#         actions_list.append(SpreadAction(HexPos(r,q), HexDir.UpRight))

class Node:
    """ A node represents a board state in the game """

    def __init__(self, board: 'GameBoard', player: PlayerColor, height, action:'SpreadAction|SpawnAction'=None):
        self.board = board
        self.player = player
        self.action = action
        # The height is the distance from the leaf_node
        self.height = height
        self.children = self.get_valid_moves(board, player)


    def eval(self) -> int:
        """
        The evaluation function. 
        """
        #TODO: write eval function here
        value = self.board.count_power(PlayerColor.RED) - self.board.count_power(PlayerColor.BLUE)
        return value
    

    def order_children(self, player):
        if player == PlayerColor.RED:
            return self.children
        else:
            return reversed(self.children)


# tip - moves should be ordered from best to worst for the player whos turn it is 
#       MAX -> order: [highest val ..... lowest val]
#       MIN -> ordeR: [lowest val ...... highest val]

    def get_valid_moves(self, game_board:'GameBoard', player:'PlayerColor'):
        """Generates a list of successors from this board state

        Args:
            player (PlayerColor): The phasing player

        Returns:
            [actions]: Valid actions
        """
        if self.height == 0:
            return
        
        # print(f"Getting valid actions for {player}")

        actions = []
        total_power = game_board.count_total_power()

        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = game_board.total_board[r][q]

                if cell.player is None:
                    if (total_power < constants.MAX_TOTAL_POWER):
                        new_game_board = GameBoard(game_board)
                        new_game_board.handle_valid_action(player, SpawnAction(HexPos(r,q)))
                        actions.append(Node(new_game_board, player.opponent, self.height - 1, SpawnAction(HexPos(r,q))))

                elif cell.player == player:
                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.Down))
                    actions.insert(0, Node(new_game_board, player.opponent, self.height - 1, SpreadAction(HexPos(r,q), HexDir.Down)))
                    
                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.DownLeft))
                    actions.insert(0, Node(new_game_board, player.opponent, self.height - 1,SpreadAction(HexPos(r,q), HexDir.DownLeft)))

                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.DownRight))
                    actions.insert(0, Node(new_game_board, player.opponent, self.height - 1, SpreadAction(HexPos(r,q), HexDir.DownRight)) )

                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.Up))
                    actions.insert(0, Node(new_game_board, player.opponent, self.height - 1, SpreadAction(HexPos(r,q), HexDir.Up)))

                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.UpLeft))
                    actions.insert(0, Node(new_game_board, player.opponent, self.height - 1, SpreadAction(HexPos(r,q), HexDir.UpLeft)))

                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.UpRight))
                    actions.insert(0, Node(new_game_board, player.opponent, self.height - 1, SpreadAction(HexPos(r,q), HexDir.UpRight)))

        return actions

class MiniMaxPruning:

    def __init__(self, initial_height=2):
        self.initial_height = initial_height
        self.max_val = np.inf
        self.min_val = -np.inf

        # this is used to store evaluated state values incase they come up again, avoids recalculation
        # self.state_evals = {}

    def run_minimax(self, player:'PlayerColor', board:'GameBoard') -> 'SpreadAction|SpawnAction':
        root = Node(board, player, self.initial_height)
        _, action = self.get_best_val(root, player, alpha=self.min_val, beta=self.max_val)
        return action
    
    def get_best_val(self, node: 'Node', player:'PlayerColor', alpha, beta):
        
        # If node is a leaf node
        if node.height == 0 or game.get_game_ended(node.board) is not None:
            return node.eval(), node.action
        
        # children = node.order_children(player)
        # MAX is playing
        if player == PlayerColor.RED:
            # print("MAX IS PLAYING")
            max_val = self.min_val
            best_action = None

            for child in node.children:
                # print(child.action)
                eval_value, _ = self.get_best_val(child, player.opponent, alpha, beta)
                if eval_value > max_val:
                    max_val = eval_value
                    best_action = child.action
                alpha = max(alpha, eval_value)
                # pruning
                if beta <= alpha:
                    break
            return max_val, best_action

        # MIN is playing
        else:
            # print("MIN IS PLAYING")
            min_val = self.max_val
            best_action = None

            for child in node.children:
                # print(child.action)
                eval_value, _ = self.get_best_val(child, player.opponent, alpha, beta)
                if eval_value < min_val:
                    min_val = eval_value
                    best_action = child.action

                beta = min(beta, eval_value)
                # pruning
                if beta <= alpha:
                    break
    
            return min_val, best_action

    
