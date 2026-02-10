from ..constants import DIRECTIONS, EMPTY
from ..metaothello import MetaOthello
from .base import UpdateRule
from .validation import is_in_board


class StandardFlankingUpdateRule(UpdateRule):
    """Standard Othello update rule that flips flanked opponent pieces."""

    @staticmethod
    def update(mo: MetaOthello, x: int, y: int) -> None:
        """Place piece and flip all flanked opponent pieces to current player's color."""
        curr_color = mo.next_color
        curr_x, curr_y = x, y
        mo.board[curr_x, curr_y] = curr_color

        for dir in DIRECTIONS:
            x, y = curr_x + dir[0], curr_y + dir[1]
            if not is_in_board(x, y) or mo.board[x, y] != -curr_color:
                continue

            while True:
                x, y = x + dir[0], y + dir[1]
                if not is_in_board(x, y) or mo.board[x, y] == EMPTY:
                    break
                if mo.board[x, y] == curr_color:
                    x, y = curr_x + dir[0], curr_y + dir[1]
                    while mo.board[x, y] == -curr_color:
                        mo.board[x, y] = curr_color
                        x, y = x + dir[0], y + dir[1]
                    break


class DeleteFlankingUpdateRule(UpdateRule):
    """Update rule that removes flanked opponent pieces instead of flipping them."""

    @staticmethod
    def update(mo: MetaOthello, x: int, y: int) -> None:
        """Place piece and remove all flanked opponent pieces from the board."""
        curr_color = mo.next_color
        curr_x, curr_y = x, y
        mo.board[curr_x, curr_y] = curr_color

        for dir in DIRECTIONS:
            x, y = curr_x + dir[0], curr_y + dir[1]
            if not is_in_board(x, y) or mo.board[x, y] != -curr_color:
                continue

            while True:
                x, y = x + dir[0], y + dir[1]
                if not is_in_board(x, y) or mo.board[x, y] == EMPTY:
                    break
                if mo.board[x, y] == curr_color:
                    x, y = curr_x + dir[0], curr_y + dir[1]
                    while mo.board[x, y] == -curr_color:
                        mo.board[x, y] = EMPTY
                        x, y = x + dir[0], y + dir[1]
                    break


class NoMiddleFlipUpdateRule(UpdateRule):
    """Update rule that only flips the first and last pieces in a flanked sequence.

    Middle pieces in the sequence remain unchanged.
    """

    @staticmethod
    def update(mo: MetaOthello, x: int, y: int) -> None:
        """Place piece and flip only the endpoints of flanked sequences."""
        curr_color = mo.next_color
        curr_x, curr_y = x, y
        mo.board[curr_x, curr_y] = curr_color

        for dir in DIRECTIONS:
            x, y = curr_x + dir[0], curr_y + dir[1]
            if not is_in_board(x, y) or mo.board[x, y] != -curr_color:
                continue

            flanked = [(x, y)]
            while True:
                x, y = x + dir[0], y + dir[1]
                if not is_in_board(x, y) or mo.board[x, y] == EMPTY:
                    break
                if mo.board[x, y] == curr_color:
                    if len(flanked) > 1:
                        to_flip = [flanked[0], flanked[-1]]
                        for x, y in to_flip:
                            mo.board[x, y] = curr_color
                    elif len(flanked) == 1:
                        x, y = flanked[0]
                        mo.board[x, y] = curr_color
                    break
                if mo.board[x, y] == -curr_color:
                    flanked.append((x, y))
