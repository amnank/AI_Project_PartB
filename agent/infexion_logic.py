from referee.game import PlayerColor, SpawnAction, HexPos, HexDir, SpreadAction, constants # pylint: disable=import-error

spread_dirs = [HexDir.Down, HexDir.DownLeft, HexDir.DownRight, HexDir.Up, HexDir.UpLeft, HexDir.UpRight]

def get_game_ended(game_board:'GameBoard') -> 'int|None':
    """Returns the final value of the game (+1, 0, -1) if ended, else None

    Returns:
        int|None: The value of the game state
    """
    if game_board.moves_played == constants.MAX_TURNS:
        red_count = game_board.count_power(PlayerColor.RED)
        blue_count = game_board.count_power(PlayerColor.BLUE)

        if red_count > blue_count and (red_count - blue_count >= constants.WIN_POWER_DIFF):
            # print("RED WON")
            return int(PlayerColor.RED)
        elif blue_count > red_count and (blue_count - red_count >= constants.WIN_POWER_DIFF):
            # print("BLUE WON")
            return int(PlayerColor.BLUE)
        else:
            # print("DRAW")
            return 0
    
    if game_board.moves_played < 2:
        return None

    red_count = 0
    blue_count = 0

    for r in range(constants.BOARD_N):
        for q in range(constants.BOARD_N):
            cell = game_board.total_board[r][q]
            if cell.player is None:
                continue
            if cell.player == PlayerColor.RED:
                red_count += 1
            elif cell.player == PlayerColor.BLUE:
                blue_count += 1

    if (red_count == 0 and blue_count == 0):
        # print("DRAW")
        return 0
    elif (red_count == 0):
        # print("BLUE WON")
        return int(PlayerColor.BLUE)
    elif (blue_count == 0):
        # print("RED WON")
        return int(PlayerColor.RED)
    else:
        return None

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

    def get_cells_under_attack(self, defender:'PlayerColor'):
        attacker = defender.opponent
        defender_cells = self.get_player_board(defender)
        attack_count = 0
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                pos = HexPos(r, q)
                if cell.player == attacker:
                    for _ in range(cell.power):
                        for spread_dir in spread_dirs:
                            spread_r, spread_q = (pos + spread_dir).r, (pos + spread_dir).q
                            if defender_cells[spread_r][spread_q] > 0:
                                attack_count += 1
                                defender_cells[spread_r][spread_q] = 0
        
        return attack_count
                    
        
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
    
    def count_cells(self, player:'PlayerColor'):
        total_cells = 0
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                if cell.player == player:
                    total_cells += 1
        
        return total_cells

class Cell:
    def __init__(self, player:'PlayerColor|None', power):
        self.player = player
        self.power = power
    
    def __str__(self):
        return f"{self.player}, POWER: {self.power}"

    def _perform_spread(self, spreading_player:'PlayerColor'):
        self.power += 1
        self.player = spreading_player

        if self.power >  constants.MAX_CELL_POWER:
            self.power = 0
            self.player = None