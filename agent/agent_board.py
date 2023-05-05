from referee.game import \
    PlayerColor, SpawnAction, SpreadAction, constants

class AgentBoard:
    def __init__(self):
        self.total_board = []
        for _ in range(constants.BOARD_N):
            col = []
            for _ in range(constants.BOARD_N):
                col.append(Cell(None, 0))
            self.total_board.append(col)

    def handle_spawn(self, player, spawn_action:'SpawnAction'):
        self.total_board[spawn_action.cell.q][spawn_action.cell.r] = Cell(player, 1)

    def handle_spread(self, player, spread_action:'SpreadAction'):
        spread_cell_power = self.total_board[spread_action.cell.q][spread_action.cell.r].power
        spread_direction = spread_action.direction
        cell = spread_action.cell

        for _ in range(spread_cell_power):
            cell += spread_direction
            self.total_board[cell.q][cell.r].perform_spread(player)

    def get_game_ended(self):
        red = False
        blue = False

        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                if cell.player == PlayerColor.RED:
                    red = True
                if cell.player == PlayerColor.BLUE:
                    blue = True

        if (red and not blue):
            return int(PlayerColor.RED)
        elif (not red and blue):
            return int(PlayerColor.BLUE)
        elif (not red and not blue):
            return 0
        else:
            # Game in progress
            return None
            

    def get_canonical_board(self, player:PlayerColor):
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

class Cell:
    def __init__(self, player:'PlayerColor|None', power):
        self.player = player
        self.power = power

    def perform_spread(self, spreading_player:'PlayerColor'):
        self.power += 1
        self.player = spreading_player

        if self.power >  6:
            self.power = 0
            self.player = None


        