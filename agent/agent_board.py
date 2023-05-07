import numpy as np
from referee.game import \
    PlayerColor, SpawnAction, HexPos, HexDir, SpreadAction, constants

class AgentBoard:
    def __init__(self):
        self.total_board = []
        self.moves_played = 0
        for _ in range(constants.BOARD_N):
            col = []
            for _ in range(constants.BOARD_N):
                col.append(Cell(None, 0))
            self.total_board.append(col)
    
    def get_valid_moves(self, player:'PlayerColor'):
        actions = []
        total_power = self.count_total_power()
        for rizz in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[rizz][q]
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
    
    def handle_action(self, player, action):
        self.moves_played += 1
        match action:
            case SpawnAction():
                self.handle_spawn(player, action)
            case SpreadAction():
                self.handle_spread(player, action)


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
        if (self.moves_played == constants.MAX_TURNS):
            return 0

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
            

    def get_canonical_board(self, player:PlayerColor):
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
    
    def count_total_power(self):
        total_power = 0
        for r in range(constants.BOARD_N):
            for q in range(constants.BOARD_N):
                cell = self.total_board[r][q]
                total_power += cell.power
        
        return total_power
    

def get_symmetries(board):
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

    def perform_spread(self, spreading_player:'PlayerColor'):
        self.power += 1
        self.player = spreading_player

        if self.power >  constants.MAX_CELL_POWER:
            self.power = 0
            self.player = None