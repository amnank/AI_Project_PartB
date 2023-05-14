import numpy as np
from .infexion_logic import InfexionGame, GameBoard # pylint: disable=import-error
from referee.game import PlayerColor, constants, SpawnAction, SpreadAction, HexPos, HexDir

game = InfexionGame()

actions_list = []
for q in range(constants.BOARD_N):
    for r in range(constants.BOARD_N):
        actions_list.append(SpawnAction(HexPos(r,q)))

for q in range(constants.BOARD_N):
    for r in range(constants.BOARD_N):
        actions_list.append(SpreadAction(HexPos(r,q), HexDir.DownLeft))
        actions_list.append(SpreadAction(HexPos(r,q), HexDir.Down))
        actions_list.append(SpreadAction(HexPos(r,q), HexDir.DownRight))
        actions_list.append(SpreadAction(HexPos(r,q), HexDir.Up))
        actions_list.append(SpreadAction(HexPos(r,q), HexDir.UpLeft))
        actions_list.append(SpreadAction(HexPos(r,q), HexDir.UpRight))

class Node:
    """ A node represents a board state in the game """

    def __init__(self, board: 'GameBoard', player: PlayerColor, height, action=None):
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
        red_value = 10 * self.board.count_power(PlayerColor.RED) +\
                    0.5 * self.board.count_cells(PlayerColor.RED)
        blue_value = 10 * self.board.count_power(PlayerColor.BLUE) +\
                    0.5 * self.board.count_cells(PlayerColor.BLUE)

        return red_value - blue_value

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

        # if game.get_game_ended(game_board) is not None:
        #     print("GAME ENDED... RETURNING")

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
                        actions.append(actions_list.index(SpawnAction(HexPos(r,q))))

                elif cell.player == player:
                    actions.insert(0, actions_list.index(SpreadAction(HexPos(r,q), HexDir.Down)))
                    actions.insert(0, actions_list.index(SpreadAction(HexPos(r,q), HexDir.DownLeft)))
                    actions.insert(0, actions_list.index(SpreadAction(HexPos(r,q), HexDir.DownRight)))
                    actions.insert(0, actions_list.index(SpreadAction(HexPos(r,q), HexDir.Up)))
                    actions.insert(0, actions_list.index(SpreadAction(HexPos(r,q), HexDir.UpLeft)))
                    actions.insert(0, actions_list.index(SpreadAction(HexPos(r,q), HexDir.UpRight)))

        return actions

class MiniMaxPruning:

    def __init__(self, initial_height=2):
        self.initial_height = initial_height
        self.max_val = np.Infinity
        self.min_val = -np.Infinity

        # this is used to store evaluated state values incase they come up again, avoids recalculation
        # self.state_evals = {}

    def run_minimax(self, player:'PlayerColor', board:'GameBoard') -> 'SpreadAction|SpawnAction':
        root = Node(board, player, self.initial_height)
        _, action = self.get_best_val(root, player, alpha=self.min_val, beta=self.max_val)
        return action
    
    def get_best_val(self, node: 'Node', player:'PlayerColor', alpha, beta):
        
        game_end = game.get_game_ended(node.board)
        if game_end is not None:
            return game_end * np.Infinity, actions_list[node.action]
        # If node is a leaf node
        if node.height == 0:
            return node.eval(), actions_list[node.action]
    
    
        # MAX is playing
        if player == PlayerColor.RED:
            # print("MAX IS PLAYING")
            max_val = self.min_val
            best_action = None
            # print(node.children)

            for _, action in enumerate(node.children):
                # print(child.action)
                board = GameBoard(node.board)
                board.handle_valid_action(player, actions_list[action])
                board.moves_played = node.board.moves_played + 1
                child = Node(board, player.opponent, node.height - 1, action)
                eval_value, _ = self.get_best_val(child, player.opponent, alpha, beta)
                if eval_value > max_val:
                    max_val = eval_value
                    best_action = child.action
                elif eval_value == self.max_val or eval_value == self.min_val:
                    max_val = eval_value
                    best_action = child.action

                alpha = max(alpha, eval_value)
                # pruning
                if beta <= alpha:
                    # print("PRUNED")
                    break
                
            return max_val, actions_list[best_action]

        # MIN is playing
        else:
            # print("MIN IS PLAYING")
            min_val = self.max_val
            best_action = None

            for _, action in enumerate(node.children):
                # print(child.action)
                board = GameBoard(node.board)
                board.handle_valid_action(player, actions_list[action])
                board.moves_played = node.board.moves_played + 1
                child = Node(board, player.opponent, node.height - 1, action)
                eval_value, _ = self.get_best_val(child, player.opponent, alpha, beta)
                if eval_value < min_val:
                    min_val = eval_value
                    best_action = child.action
                elif eval_value == self.max_val or eval_value == self.min_val:
                    min_val = eval_value
                    best_action = child.action

                beta = min(beta, eval_value)
                # pruning
                if beta <= alpha:
                    # print("PRUNED")
                    break
    
            return min_val, actions_list[best_action]

    
