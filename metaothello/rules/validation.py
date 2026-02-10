from ..constants import BOARD_DIM, DIRECTIONS, EMPTY
from ..metaothello import MetaOthello
from .base import ValidationRule


def is_in_board(x: int, y: int) -> bool:
    """Check if coordinates are within the board boundaries."""
    return 0 <= x < BOARD_DIM and 0 <= y < BOARD_DIM


class AvailableRule(ValidationRule):
    """Checks if the move is made on an empty square."""

    @staticmethod
    def is_valid(mo: MetaOthello, x: int, y: int) -> bool:
        """Check if the square at (x, y) is empty."""
        return mo.board[x, y] == EMPTY


class StandardFlankingValidationRule(ValidationRule):
    """Checks if the move captures at least one opponent piece in any direction."""

    @staticmethod
    def is_valid(mo: MetaOthello, x: int, y: int) -> bool:
        """Check if the move flanks at least one opponent piece."""
        curr_color = mo.next_color
        curr_x, curr_y = x, y
        for dir in DIRECTIONS:
            x, y = curr_x + dir[0], curr_y + dir[1]
            if not is_in_board(x, y) or mo.board[x, y] != -curr_color:
                continue

            while True:
                x, y = x + dir[0], y + dir[1]
                if not is_in_board(x, y) or mo.board[x, y] == EMPTY:
                    break
                if mo.board[x, y] == curr_color:
                    return True

        return False


class NeighborValidationRule(ValidationRule):
    """Checks if the move is adjacent to at least one piece of the same color."""

    @staticmethod
    def is_valid(mo: MetaOthello, x: int, y: int) -> bool:
        """Check if the move is adjacent to a piece of the same color."""
        curr_color = mo.next_color
        curr_x, curr_y = x, y
        for dir in DIRECTIONS:
            x, y = curr_x + dir[0], curr_y + dir[1]
            if is_in_board(x, y) and mo.board[x, y] == curr_color:
                return True

        return False
