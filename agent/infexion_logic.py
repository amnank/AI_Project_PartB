from referee.game import PlayerColor, SpawnAction, HexPos, HexDir, SpreadAction, constants # pylint: disable=import-error

spread_dirs = [HexDir.Down,
               HexDir.DownLeft,
               HexDir.DownRight,
               HexDir.Up,
               HexDir.UpLeft,
               HexDir.UpRight]

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
            self.total_board[cell.r][cell.q].perform_spread(player)
        
        self.total_board[spread_action.cell.r][spread_action.cell.q] = Cell(None, 0)

    def max_cells_under_attack(self, defender:'PlayerColor'):
        """This function returns the maximum cells under attack by one spread action

        Args:
            defender (PlayerColor): The defending player

        Returns:
            int: Max cells under attack count
        """
        attacker = defender.opponent
        defender_cells = self.get_player_board(defender)
        cell_max = 0
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                pos = HexPos(r, q)
                if cell.player == attacker:
                    for spread_dir in spread_dirs:
                        dir_max = 0
                        for _ in range(cell.power):
                            dir_count = 0
                            spread_r, spread_q = (pos + spread_dir).r, (pos + spread_dir).q
                            if defender_cells[spread_r][spread_q] > 0:
                                defender_cells[spread_r][spread_q] = 0
                                dir_count += 1
                        dir_max = max(dir_count, dir_max)
                    cell_max = max(cell_max, dir_max)
        
        return cell_max
    
    def get_empty_cells_under_attack(self, player:'PlayerColor'):
        """This function returns the count of empty cells that are "covered" by
        possible spread actions by a particular player's cells

        Args:
            player (PlayerColor): The attacking player

        Returns:
            int: Count of covered empty cells
        """
        empty_cells = self.get_empty_spaces()
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                pos = HexPos(r, q)
                if cell.player == player:
                    for spread_dir in spread_dirs:
                        for _ in range(cell.power):
                            spread_r, spread_q = (pos + spread_dir).r, (pos + spread_dir).q
                            if empty_cells[spread_r][spread_q] > 0:
                                empty_cells[spread_r][spread_q] = 0
        attack_count = 0
        for r in empty_cells:
            for c in r:
                if c == 0:
                    attack_count += 1
        return attack_count
    
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
    
    def count_power(self, player:'PlayerColor'):
        """This function returns the total power on the board
        belonging to a player

        Args:
            player (PlayerColor): The player

        Returns:
            int: Power count
        """
        total_power = 0
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                if cell.player == player:
                    total_power += cell.power
        
        return total_power

    def count_total_power(self):
        """This function returns the total power on the board

        Returns:
            int: Total power
        """
        total_power = 0
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                total_power += cell.power
        
        return total_power
    
    def count_cells(self, player:'PlayerColor'):
        """This function counts the total number of cells belonging to one player

        Args:
            player (PlayerColor): The player

        Returns:
            int: Cells count
        """
        total_cells = 0
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                if cell.player == player:
                    total_cells += 1
       
        return total_cells

class Cell:
    """This class encapsulates the logic of a single cell on the game board
    """
    def __init__(self, player:'PlayerColor|None', power):
        self.player = player
        self.power = power
    
    def __str__(self):
        return f"{self.player}, POWER: {self.power}"

    def perform_spread(self, spreading_player:'PlayerColor'):
        """This function performs the effect of a spread action on the cell

        Args:
            spreading_player (PlayerColor): The attacking player
        """
        self.power += 1
        self.player = spreading_player

        if self.power >  constants.MAX_CELL_POWER:
            self.power = 0
            self.player = None