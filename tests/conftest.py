"""Pytest configuration and fixtures for metaothello tests."""

import numpy as np
import pytest

from metaothello.constants import BLACK, EMPTY, WHITE
from metaothello.games import ClassicOthello, DeleteFlanking, Iago, NoMiddleFlip


@pytest.fixture
def classic_game() -> ClassicOthello:
    """Fresh ClassicOthello instance."""
    return ClassicOthello()


@pytest.fixture
def nomidflip_game() -> NoMiddleFlip:
    """Fresh NoMiddleFlip instance."""
    return NoMiddleFlip()


@pytest.fixture
def delflank_game() -> DeleteFlanking:
    """Fresh DeleteFlanking instance."""
    return DeleteFlanking()


@pytest.fixture
def iago_game() -> Iago:
    """Fresh Iago instance."""
    return Iago()


@pytest.fixture
def classic_game_with_moves() -> ClassicOthello:
    """ClassicOthello after a known opening sequence."""
    game = ClassicOthello()
    game.play_move("d3")  # BLACK
    game.play_move("c3")  # WHITE
    game.play_move("b3")  # BLACK
    return game


@pytest.fixture
def full_board_game() -> ClassicOthello:
    """ClassicOthello with a nearly full board (for testing passes)."""
    game = ClassicOthello()
    # Fill board with BLACK pieces except one position
    game.board = np.full((8, 8), BLACK)
    game.board[0, 0] = EMPTY
    game.next_color = WHITE
    return game
