import sys
sys.path.append("game")
import numpy as np
from infexion_logic import infexion_game, GameBoard # pylint: disable=import-error
from game import PlayerColor, SpawnAction, HexPos, HexDir, SpreadAction, constants # pylint: disable=import-error

policy_actions = []
for q in range(constants.BOARD_N):
    for r in range(constants.BOARD_N):
        policy_actions.append(SpawnAction(HexPos(r,q)))

for q in range(constants.BOARD_N):
    for r in range(constants.BOARD_N):
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.DownLeft))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.Down))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.DownRight))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.Up))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.UpLeft))
        policy_actions.append(SpreadAction(HexPos(r,q), HexDir.UpRight))

policy_actions = np.array(policy_actions)

def normalize_policy(policy):
    policy = np.array(policy)
    policy = (policy / policy.sum()).flatten()
    return policy

def valid_action_mask(game_board:'GameBoard', player:'PlayerColor'):
        """This function creates a 343 x 1 vector corresponding to
        alpha_zero_helper.policy_actions with all the valid moves

        Args:
            player (PlayerColor): The phasing player

        Returns:
            343 x 1 list: One hot encoded valid moves
        """
        mask = []
        for action in policy_actions:
            if infexion_game.is_valid_move(game_board, player, action):
                mask.append(1)
            else:
                mask.append(0)

        return mask

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
        np.array(): List of boards
    """
    inp = []

    inp.append(board.get_canonical_board(player))
    inp.append(board.get_empty_spaces())
    inp.append(board.get_player_board(player))
    inp.append(board.get_player_board(player.opponent))

    for power in range(1, constants.MAX_CELL_POWER):
        inp.append(board.get_player_power_board(power, player))
        inp.append(_reverse_board_sign(board.get_player_power_board(power, player.opponent)))

    return np.array(inp)

def sample_policy(policy) -> 'SpawnAction|SpreadAction':
    """This function samples an action from the policy

    Args:
        policy (343 x 1 List): probabilities

    Returns:
        SpawnAction | Spread Action: The sampled action
    """
    policy = np.array(policy)
    policy = (policy / policy.sum()).flatten()
    action = np.random.choice(np.array(policy_actions), p=policy)
    return action

def greedy_select_from_policy(policy) -> 'SpawnAction|SpreadAction':
    """This function greedy selects an action from the policy

    Args:
        policy (343 x 1 list): probabilities

    Returns:
        SpawnAction|SpreadAction: The selected action
    """
    policy = normalize_policy(policy)
    action = policy_actions[int(np.array(policy).argmax())]
    return action

def _reverse_board_sign(board):
    return [[-i for i in row] for row in board]

def get_policy_symmetries(policy):
    """This function returns the list of symmetries of a policy vector,
    to ensure that it stays consistent with rotations of the input board

    Args:
        policy (343 x 1 list): Policies

    Returns:
        list(policies): List of symmetries
    """
    spawn_policy = policy[:49]
    spread_policy = policy[49:]

    spawn_array = np.array(spawn_policy)
    spread_array = np.array(spread_policy)

    spawn_array = np.reshape(spawn_array, (7, 7))
    spread_array = np.reshape(spread_array, (7, 7, 6))

    spawn_symmetries = get_symmetries(spawn_array.tolist())
    spread_symmetries = get_symmetries(spread_array.tolist())

    output = []
    for spawn_sym, spread_sym in zip(spawn_symmetries, spread_symmetries):
        spa = np.array(spawn_sym).flatten().tolist()
        spr = np.array(spread_sym).flatten().tolist()
        output.append(spa + spr)

    return np.array(output).reshape((3,343,1))
    
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

    # Add 90 degree clockwise rotation
    symmetries.append(np.rot90(board, k=-1))
    # Add 180 degree clockwise rotation
    symmetries.append(np.rot90(board, k=-2))
    # Add 270 degree clockwise rotation
    symmetries.append(np.rot90(board, k=-3))
    # Add horizontal flip
    # symmetries.append(np.fliplr(board))
    # # Add vertical flip
    # symmetries.append(np.flipud(board))
    # # Add diagonal flip
    # symmetries.append(np.fliplr(np.rot90(board)))
    # # Add anti-diagonal flip
    # symmetries.append(np.rot90(np.fliplr(board)))

    return symmetries
