import numpy as np
from alpha_zero_helper import policy_actions # pylint: disable=import-error
from referee.game import \
    PlayerColor, SpawnAction, HexPos, HexDir, SpreadAction, constants

class AgentGame:
    """The class encapsulates the logic of the internal representation of the
    Agent's game state
    """
    def __init__(self):
        self._total_board = []
        self.moves_played = 0
        for _ in range(constants.BOARD_N):
            col = []
            for _ in range(constants.BOARD_N):
                col.append(Cell(None, 0))
            self._total_board.append(col)
    
    def get_valid_moves(self, player:'PlayerColor'):
        """Generates a list of valid action from this board state

        Args:
            player (PlayerColor): The phasing player

        Returns:
            [actions]: Valid actions
        """
        actions = []
        total_power = self.count_total_power()
        for rizz in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self._total_board[rizz][q]
                if cell.player is None:
                    if (total_power < constants.MAX_TOTAL_POWER):
                        actions.append(SpawnAction(HexPos(rizz,q)))
                elif cell.player == player:
                    actions.append(SpreadAction(HexPos(rizz,q), HexDir.Down))
                    actions.append(SpreadAction(HexPos(rizz,q), HexDir.DownLeft))
                    actions.append(SpreadAction(HexPos(rizz,q), HexDir.DownRight))
                    actions.append(SpreadAction(HexPos(rizz,q), HexDir.Up))
                    actions.append(SpreadAction(HexPos(rizz,q), HexDir.UpLeft))
                    actions.append(SpreadAction(HexPos(rizz,q), HexDir.UpRight))
        
        return actions
    
    def valid_action_mask(self, player:'PlayerColor'):
        """This function creates a 343 x 1 vector corresponding to
        alpha_zero_helper.policy_actions with all the valid moves

        Args:
            player (PlayerColor): The phasing player

        Returns:
            343 x 1 list: One hot encoded valid moves
        """
        mask = []
        for action in policy_actions:
            if self._is_valid_move(player, action):
                mask.append(1)
            else:
                mask.append(0)

        return mask
    
    def _is_valid_move(self, player, action) -> bool:
        """Returns whether an action made by a player is valid or not

        Args:
            player (PlayerColor): The phasing player
            action (SpawnAction | SpreadAction): The action

        Returns:
            bool: is_valid
        """
        cell = self._total_board[action.cell.r][action.cell.q]
        total_power = self.count_total_power()
        
        match action:
            case SpawnAction():
                if cell.player is None and (total_power < constants.MAX_TOTAL_POWER):
                    return True
                else:
                    return False
            case SpreadAction():
                if cell.player == player:
                    return True
                else:
                    return False
    
    def handle_valid_action(self, player, action):
        """Handles the effects of a valid action on the board

        Args:
            player (PlayerColor): The player making the action
            action (SpawnAction | SpreadAction): The action
        """
        self.moves_played += 1
        match action:
            case SpawnAction():
                self._handle_spawn(player, action)
            case SpreadAction():
                self._handle_spread(player, action)


    def _handle_spawn(self, player, spawn_action:'SpawnAction'):
        self._total_board[spawn_action.cell.q][spawn_action.cell.r] = Cell(player, 1)

    def _handle_spread(self, player, spread_action:'SpreadAction'):
        spread_cell_power = self._total_board[spread_action.cell.q][spread_action.cell.r].power
        spread_direction = spread_action.direction
        cell = spread_action.cell

        for _ in range(spread_cell_power):
            cell += spread_direction
            self._total_board[cell.q][cell.r].perform_spread(player)
        
        spread_action.cell = Cell(None, 0)

    def get_game_ended(self) -> 'int|None':
        """Returns the final value of the game (+1, 0, -1) if ended, else None

        Returns:
            int|None: The value of the game state
        """
        if (self.moves_played == constants.MAX_TURNS):
            return 0

        red = False
        blue = False

        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self._total_board[r][q]
                if cell.player == PlayerColor.RED:
                    red = True
                if cell.player == PlayerColor.BLUE:
                    blue = True

        if (red and not blue):
            # Red win
            return int(PlayerColor.RED)
        elif (not red and blue):
            # Blue win
            return int(PlayerColor.BLUE)
        elif (not red and not blue):
            # Draw
            return 0
        else:
            # Game in progress
            return None
            

    def get_canonical_board(self, player:PlayerColor) -> list(int):
        """Returns the board with player cells having +ve power,
        and opponent cells having -ve power

        Args:
            player (PlayerColor): The phasing player

        Returns:
            list(int): The board
        """
        board = []
        for r in range(constants.BOARD_N):
            col = []
            for q in range(constants.BOARD_N):
                cell = self._total_board[r][q]
                if cell.player == player:
                    col.append(cell.power)
                elif cell.player == player.opponent:
                    col.append(-1 * cell.power)
                else:
                    col.append(0)
            board.append(col)

        return board
    
    def get_player_board(self, player:PlayerColor) -> list(int):
        """Returns the board with only the phasing player's cells, all other
        cells are 0

        Args:
            player (PlayerColor): The phasing player

        Returns:
            list(int): The board
        """
        board = []
        for r in range(constants.BOARD_N):
            col = []
            for q in range(constants.BOARD_N):
                cell = self._total_board[r][q]
                if cell.player == player:
                    col.append(cell.power)
                else:
                    col.append(0)
            board.append(col)

        return board
    
    def get_empty_spaces(self) -> list(int):
        """Returns the board with empty cells having a value of 1, else 0

        Returns:
            list(int): The board
        """
        board = []
        for r in range(constants.BOARD_N):
            col = []
            for q in range(constants.BOARD_N):
                cell = self._total_board[r][q]
                if cell.player is None:
                    col.append(1)
                else:
                    col.append(0)
            board.append(col)

        return board
    
    def get_player_power_board(self, power, player:'PlayerColor') -> list(int):
        """Returns the board where only cells having a particular power of the
        phasing player are included, else 0

        Args:
            power (int): The power to be included
            player (PlayerColor): The phasing player
            
        Returns:
            list(int): Board
        """
        board = []
        for r in range(constants.BOARD_N):
            col = []
            for q in range(constants.BOARD_N):
                cell = self._total_board[r][q]
                if cell.player == player and cell.power == power:
                    col.append(cell.power)
                else:
                    col.append(0)
            board.append(col)

        return board
    
    def count_total_power(self) -> int:
        """Returns the total power of all cells on the board

        Returns:
            int: The total power
        """
        total_power = 0
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self._total_board[r][q]
                total_power += cell.power
        
        return total_power
    

def get_symmetries(board):
    """Takes a board state and generates a list containing the 90 degree, 180 degree
    and 270 degree rotations, as well as the horizontal, vertical, diagonal and anti-diagonal
    flips.

    Args:
        board (list(int)): The board state

    Returns:
        list(list(int)): List of board states
    """
    symmetries = []
    symmetries.append(board)

    brd = np.array(board)

    # Add 90 degree clockwise rotation
    symmetries.append(np.rot90(brd, k=-1).tolist())

    # Add 180 degree clockwise rotation
    symmetries.append(np.rot90(brd, k=-2).tolist())
    
    # Add 270 degree clockwise rotation
    symmetries.append(np.rot90(brd, k=-3).tolist())
    
    # Add horizontal flip
    symmetries.append(np.fliplr(brd).tolist())
    
    # Add vertical flip
    symmetries.append(np.flipud(brd).tolist())
    
    # Add diagonal flip
    symmetries.append(np.fliplr(np.rot90(brd)).tolist())
    
    # Add anti-diagonal flip
    symmetries.append(np.rot90(np.fliplr(brd)).tolist())

    return symmetries


class Cell:
    def __init__(self, player:'PlayerColor|None', power):
        self.player = player
        self.power = power

    def _perform_spread(self, spreading_player:'PlayerColor'):
        self.power += 1
        self.player = spreading_player

        if self.power >  constants.MAX_CELL_POWER:
            self.power = 0
            self.player = None