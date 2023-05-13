import numpy as np
from infexion_logic import InfexionGame, GameBoard # pylint: disable=import-error
from referee.game import PlayerColor, constants, SpawnAction, SpreadAction, HexPos, HexDir

game = InfexionGame()

class Node:
    """ A node represents a board state in the game """

    def __init__(self, board: 'GameBoard', player: PlayerColor, height):
        self.value = self.eval()
        self.board = board
        self.player = player
        self.children = game.get_valid_moves(board, player)

        # The height is the distance from the leaf_node
        self.height = height


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
        actions = []
        total_power = game_board.count_total_power()

        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = game_board[r][q]

                if cell.player is None:
                    if (total_power < constants.MAX_TOTAL_POWER):
                        new_game_board = GameBoard(game_board)
                        new_game_board.handle_valid_action(player, SpawnAction(HexPos(r,q)))
                        actions.append(Node(new_game_board, player, self.height - 1))

                elif cell.player == player:
                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.Down))
                    actions.insert(0, Node(new_game_board, player, self.height - 1))
                    
                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.DownLeft))
                    actions.insert(0, Node(new_game_board, player, self.height - 1))

                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.DownRight))
                    actions.insert(0, Node(new_game_board, player, self.height - 1) )

                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.Up))
                    actions.insert(0, Node(new_game_board, player, self.height - 1))

                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.UpLeft))
                    actions.insert(0, Node(new_game_board, player, self.height - 1))

                    new_game_board = GameBoard(game_board)
                    new_game_board.handle_valid_action(player, SpreadAction(HexPos(r,q), HexDir.UpRight))
                    actions.insert(0, Node(new_game_board, player, self.height - 1))

        return actions

class MiniMaxPruning:

    def __init__(self):
        self.game = game
        self.player = PlayerColor.RED

        self.initial_height = 20
        self.max_val = np.inf
        self.min_val = -np.inf

        self.root = Node(GameBoard(), self.player, self.initial_height)

        # this is used to store evaluated state values incase they come up again, avoids recalculation
        # self.state_evals = {}

    def run_minimax(self):
        self.get_best_val(self.root, self.player, alpha=self.min_val, beta=self.max_val)
    
    def get_best_val(self, node: 'Node', player:'PlayerColor', alpha, beta):
        
        # If node is a leaf node
        if node.height == 0 or game.get_game_ended(node.board) is not None:
            return node.eval()
        
        children = node.order_children(player)
        # MAX is playing
        if player == PlayerColor.RED:
            max_val = self.min_val

            for child in node.children:
                eval_value = self.get_best_val(child, player.opponent, alpha, beta)
                max_val = max(max_val, eval_value)
                alpha = max(alpha, eval_value)

                # pruning
                if beta <= alpha:
                    break
            

            return max_val

        # MIN is playing
        else:
            min_val = self.max_val

            for child in children:
                eval_value = self.get_best_val(child, player.opponent, alpha, beta)
                min_val = min(min_val, eval_value)
                beta = min(beta, eval_value)

                # pruning
                if beta <= alpha:
                    break
    
            return min_val

    
