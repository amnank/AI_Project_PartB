import numpy as np
from agent_game import AgentGame # pylint: disable=import-error
from referee.game import \
    PlayerColor, SpawnAction, HexPos, HexDir, SpreadAction, constants

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



def create_input(player:PlayerColor, board:AgentGame):
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