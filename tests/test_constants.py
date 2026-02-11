"""Tests for constants and coordinate mappings."""

import re

from metaothello.constants import (
    BLACK,
    BOARD_DIM,
    DIRECTIONS,
    EMPTY,
    MAX_STEPS,
    SQUARES,
    WHITE,
    move2tuple,
    tuple2move,
)


class TestConstants:
    """Test game constants and coordinate mappings."""

    def test_board_dimensions(self) -> None:
        """Verify BOARD_DIM == 8, MAX_STEPS == 60."""
        assert BOARD_DIM == 8
        assert MAX_STEPS == 60

    def test_player_colors(self) -> None:
        """Verify BLACK == -1, WHITE == 1, EMPTY == 0, BLACK == -WHITE."""
        assert BLACK == -1
        assert WHITE == 1
        assert EMPTY == 0
        assert BLACK == -WHITE

    def test_directions_count(self) -> None:
        """Verify len(DIRECTIONS) == 8, no [0,0] entry."""
        assert len(DIRECTIONS) == 8
        assert [0, 0] not in DIRECTIONS

    def test_directions_completeness(self) -> None:
        """Verify all 8 compass directions are present."""
        expected_directions = [
            [-1, -1],  # NW
            [-1, 0],  # N
            [-1, 1],  # NE
            [0, -1],  # W
            [0, 1],  # E
            [1, -1],  # SW
            [1, 0],  # S
            [1, 1],  # SE
        ]
        for expected_dir in expected_directions:
            assert expected_dir in DIRECTIONS

    def test_squares_count(self) -> None:
        """Verify len(SQUARES) == 64."""
        assert len(SQUARES) == 64

    def test_squares_format(self) -> None:
        """Verify each square matches pattern [a-h][1-8]."""
        pattern = re.compile(r"^[a-h][1-8]$")
        for square in SQUARES:
            assert pattern.match(square), f"Invalid square format: {square}"

    def test_move2tuple_completeness(self) -> None:
        """Verify 64 entries, all values are (row, col) tuples in range [0,7]."""
        assert len(move2tuple) == 64
        for move, (row, col) in move2tuple.items():
            assert 0 <= row < 8, f"Invalid row {row} for move {move}"
            assert 0 <= col < 8, f"Invalid col {col} for move {move}"

    def test_tuple2move_completeness(self) -> None:
        """Verify 64 entries, all values are valid move strings."""
        assert len(tuple2move) == 64
        pattern = re.compile(r"^[a-h][1-8]$")
        for _coords, move in tuple2move.items():
            assert pattern.match(move), f"Invalid move string: {move}"

    def test_move2tuple_tuple2move_roundtrip(self) -> None:
        """For every square: tuple2move[move2tuple[s]] == s and vice versa."""
        # Test move -> tuple -> move
        for square in SQUARES:
            coords = move2tuple[square]
            recovered_square = tuple2move[coords]
            assert recovered_square == square, f"Roundtrip failed for {square}"

        # Test tuple -> move -> tuple
        for coords, move in tuple2move.items():
            recovered_coords = move2tuple[move]
            assert recovered_coords == coords, f"Roundtrip failed for {coords}"

    def test_specific_coordinate_mappings(self) -> None:
        """Verify known mappings: 'a1' -> (0,0), 'h8' -> (7,7), 'd3' -> (2,3), 'e4' -> (3,4)."""
        assert move2tuple["a1"] == (0, 0)
        assert move2tuple["h8"] == (7, 7)
        assert move2tuple["d3"] == (2, 3)
        assert move2tuple["e4"] == (3, 4)

        assert tuple2move[(0, 0)] == "a1"
        assert tuple2move[(7, 7)] == "h8"
        assert tuple2move[(2, 3)] == "d3"
        assert tuple2move[(3, 4)] == "e4"
