import sys
sys.path.append("game")
from game import PlayerColor, SpawnAction, HexPos, HexDir, SpreadAction, constants # pylint: disable=import-error

class InfexionGame:
    """The class encapsulates the logic of the Infexion game
    """
    
    def get_valid_moves(self, game_board:'GameBoard', player:'PlayerColor'):
        """Generates a list of valid action from this board state

        Args:
            player (PlayerColor): The phasing player

        Returns:
            [actions]: Valid actions
        """
        actions = []
        total_power = game_board.count_total_power()
        for rizz in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = game_board[rizz][q]
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
    
    def is_valid_move(self, game_board:'GameBoard', player, action) -> bool:
        """Returns whether an action made by a player is valid or not

        Args:
            player (PlayerColor): The phasing player
            action (SpawnAction | SpreadAction): The action

        Returns:
            bool: is_valid
        """
        cell = game_board.total_board[action.cell.r][action.cell.q]
        total_power = game_board.count_total_power()
        
        match action:
            case SpawnAction():
                if (cell.player is None) and (total_power < constants.MAX_TOTAL_POWER):
                    return True
                else:
                    return False
            case SpreadAction():
                if cell.player == player:
                    return True
                else:
                    return False
    

    def get_game_ended(self, game_board:'GameBoard') -> 'int|None':
        """Returns the final value of the game (+1, 0, -1) if ended, else None

        Returns:
            int|None: The value of the game state
        """
        if game_board.moves_played == constants.MAX_TURNS:
            red_count = game_board.count_power(PlayerColor.RED)
            blue_count = game_board.count_power(PlayerColor.BLUE)

            if red_count > blue_count and (red_count - blue_count > constants.WIN_POWER_DIFF):
                return int(PlayerColor.RED)
            elif blue_count > red_count and (blue_count - red_count > constants.WIN_POWER_DIFF):
                return int(PlayerColor.BLUE)
            else:
                return 0
        
        if game_board.moves_played < 2:
            return None

        red = False
        blue = False

        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = game_board.total_board[r][q]
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

infexion_game = InfexionGame()

class GameBoard:
    """This class encapsulates the logic of an Infexion game state
    """
    def __init__(self, other_game_board:'GameBoard|None'=None):
        self.moves_played = 0
        if other_game_board is None:
            self.total_board = []
            for _ in range(constants.BOARD_N):
                col = []
                for _ in range(constants.BOARD_N):
                    col.append(Cell(None, 0))
                self.total_board.append(col)
        else:
            self.total_board = []
            for r in range(constants.BOARD_N):
                col = []
                for q in range(constants.BOARD_N):
                    cell = other_game_board.total_board[r][q]
                    col.append(Cell(cell.player, cell.power))
                self.total_board.append(col)

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
        self.total_board[spawn_action.cell.r][spawn_action.cell.q] = Cell(player, 1)

    def _handle_spread(self, player, spread_action:'SpreadAction'):
        spread_cell_power = self.total_board[spread_action.cell.r][spread_action.cell.q].power
        spread_direction = spread_action.direction
        cell = spread_action.cell

        for _ in range(spread_cell_power):
            cell += spread_direction
            self.total_board[cell.r][cell.q]._perform_spread(player)
        
        self.total_board[spread_action.cell.r][spread_action.cell.q] = Cell(None, 0)
        
    def get_canonical_board(self, player:PlayerColor):
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
                cell = self.total_board[r][q]
                if cell.player == player:
                    col.append(cell.power)
                elif cell.player == player.opponent:
                    col.append(-1 * cell.power)
                else:
                    col.append(0)
            board.append(col)

        return board
    
    def get_player_board(self, player:PlayerColor):
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
                cell = self.total_board[r][q]
                if cell.player == player:
                    col.append(cell.power)
                else:
                    col.append(0)
            board.append(col)

        return board
    
    def get_empty_spaces(self):
        """Returns the board with empty cells having a value of 1, else 0

        Returns:
            list(int): The board
        """
        board = []
        for r in range(constants.BOARD_N):
            col = []
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                if cell.player is None:
                    col.append(1)
                else:
                    col.append(0)
            board.append(col)

        return board
    
    def get_player_power_board(self, power, player:'PlayerColor'):
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
                cell = self.total_board[r][q]
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
                cell = self.total_board[r][q]
                total_power += cell.power
        
        return total_power
    
    def count_power(self, player:'PlayerColor'):
        total_power = 0
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                if cell.player == player:
                    total_power += cell.power
        
        return total_power

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