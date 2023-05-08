import numpy as np
from referee.game import \
    PlayerColor, SpawnAction, HexPos, HexDir, SpreadAction, constants
from infexion_logic import InfexionGame, GameBoard # pylint: disable=import-error

policy_actions = []
for q in range(constants.BOARD_N):
    for r in range(constants.BOARD_N):
        policy_actions.append(SpawnAction(HexPos(r,q)))

for q in range(constants.BOARD_N):
    for r in range(constants.BOARD_N):
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.Down))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.DownLeft))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.DownRight))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.Up))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.UpLeft))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.UpRight))



def create_input(player:PlayerColor, board:'GameBoard'):
    """This function creates input for the neural network
        1. The canonical board (+self, -opponent)
        2. The empty spaces
        3. Only self pieces
        4. Only opponent pieces
        5-10. self pieces of power (1-6)
        11-16. opponent pieces of power (1-6) -ve

    Args:
        player (PlayerColor): The phasing player

    Returns:
        list(list(int)): List of boards
    """
    inp = []

    inp.append(board.get_canonical_board)
    inp.append(board.get_empty_spaces)
    inp.append(board.get_player_board(player))
    inp.append(board.get_player_board(player.oppponent))

    for power in range(1, constants.MAX_CELL_POWER):
        inp.append(board.get_player_power_board(power, player))
        inp.append(_reverse_board_sign(board.get_player_power_board(power, player.opponent)))

    return inp

def sample_policy(policy) -> 'SpawnAction|SpreadAction':
    """This function returns an action from the policy

    Args:
        policy (343 x 1 List): probabilities

    Returns:
        SpawnAction | Spread Action: The sampled action
    """
    action = np.random.choice(policy_actions, p=policy)
    return action

def _reverse_board_sign(board):
    return [[-i for i in row] for row in board]

def get_symmetries(board):
    """Takes a board list state and generates a list containing the 90 degree, 180 degree
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
