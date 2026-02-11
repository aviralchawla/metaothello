"""Comprehensive tests for the core MetaOthello class."""

import io
import sys

import numpy as np
import pytest

from metaothello.constants import BLACK, BOARD_DIM, EMPTY, MAX_STEPS, WHITE
from metaothello.games import ClassicOthello
from metaothello.metaothello import MetaOthello
from metaothello.rules.initialization import ClassicInitialization
from metaothello.rules.update import StandardFlankingUpdateRule
from metaothello.rules.validation import AvailableRule, StandardFlankingValidationRule


class TestMetaOthelloInit:
    """Test MetaOthello initialization."""

    def test_board_initialized(self) -> None:
        """Board is set up by initialization rule."""
        game = MetaOthello(ClassicInitialization, [], [])
        # Check that board is not all zeros (it has been initialized)
        assert not np.all(game.board == 0)
        assert game.board.shape == (BOARD_DIM, BOARD_DIM)

    def test_initial_state(self) -> None:
        """done=False, next_color=BLACK, history=[], board_history=[], valid_moves=None."""
        game = ClassicOthello()
        assert game.done is False
        assert game.next_color == BLACK
        assert game.history == []
        assert game.board_history == []
        assert game.valid_moves is None

    def test_custom_rules_accepted(self) -> None:
        """MetaOthello can be instantiated with any valid rule combination."""
        game = MetaOthello(
            initialization_rule=ClassicInitialization,
            validation_rules=[AvailableRule, StandardFlankingValidationRule],
            update_rules=[StandardFlankingUpdateRule],
        )
        assert game.board is not None
        assert game.initialization_rule == ClassicInitialization
        assert game.validation_rules == [AvailableRule, StandardFlankingValidationRule]
        assert game.update_rules == [StandardFlankingUpdateRule]


# ===== TestPlayMove =====


class TestPlayMove:
    """Test play_move method."""

    def test_valid_move_updates_board(self, classic_game: ClassicOthello) -> None:
        """Board state changes after valid move."""
        initial_board = classic_game.board.copy()
        classic_game.play_move("d3")
        assert not np.array_equal(classic_game.board, initial_board)
        # Check that the piece was placed
        assert classic_game.board[2, 3] == BLACK

    def test_valid_move_switches_player(self, classic_game: ClassicOthello) -> None:
        """next_color toggles from BLACK to WHITE."""
        assert classic_game.next_color == BLACK
        classic_game.play_move("d3")
        assert classic_game.next_color == WHITE
        classic_game.play_move("c3")
        assert classic_game.next_color == BLACK

    def test_move_appended_to_history(self, classic_game: ClassicOthello) -> None:
        """History grows by 1 after each move."""
        assert len(classic_game.history) == 0
        classic_game.play_move("d3")
        assert len(classic_game.history) == 1
        assert classic_game.history[0] == "d3"
        classic_game.play_move("c3")
        assert len(classic_game.history) == 2
        assert classic_game.history[1] == "c3"

    def test_board_state_appended_to_board_history(self, classic_game: ClassicOthello) -> None:
        """board_history grows by 1, contains board snapshot."""
        assert len(classic_game.board_history) == 0
        classic_game.play_move("d3")
        assert len(classic_game.board_history) == 1
        assert isinstance(classic_game.board_history[0], np.ndarray)
        assert classic_game.board_history[0].shape == (BOARD_DIM, BOARD_DIM)

    def test_board_history_is_copy(self, classic_game: ClassicOthello) -> None:
        """Modifying returned board_history list doesn't affect game state."""
        classic_game.play_move("d3")
        board_history = classic_game.get_board_history()
        # Append to the returned list
        board_history.append(np.zeros((BOARD_DIM, BOARD_DIM)))
        # Game's board_history list length should be unchanged
        assert len(classic_game.board_history) == 1
        # Note: The arrays inside are still references (not deep copies)

    def test_valid_moves_cache_cleared_after_move(self, classic_game: ClassicOthello) -> None:
        """valid_moves reset to None after play_move."""
        # Populate the cache
        classic_game.get_all_valid_moves()
        assert classic_game.valid_moves is not None
        # Play a move
        classic_game.play_move("d3")
        # Cache should be cleared
        assert classic_game.valid_moves is None

    def test_invalid_move_raises_valueerror(self, classic_game: ClassicOthello) -> None:
        """Playing an illegal move raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Move a1 is invalid"):
            classic_game.play_move("a1")  # Invalid move at start

    def test_invalid_move_error_message_contents(self, classic_game: ClassicOthello) -> None:
        """Error message includes move, game alias, history, valid moves."""
        try:
            classic_game.play_move("a1")
        except ValueError as e:
            error_msg = str(e)
            assert "a1" in error_msg
            assert "classic" in error_msg
            assert "sequence" in error_msg or "history" in error_msg

    def test_override_bypasses_validation(self, classic_game: ClassicOthello) -> None:
        """play_move(move, override=True) succeeds even for invalid move."""
        # a1 is not a valid opening move, but override should allow it
        classic_game.play_move("a1", override=True)
        assert classic_game.board[0, 0] == BLACK
        assert len(classic_game.history) == 1

    def test_pass_move_none(self) -> None:
        """play_move(None) when only valid move is pass."""
        game = ClassicOthello()
        # Set up board where BLACK must pass
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = EMPTY
        game.next_color = WHITE
        # WHITE should only be able to pass
        valid_moves = game.get_all_valid_moves()
        assert valid_moves == [None]
        # Should not raise an error
        game.play_move(None)
        assert len(game.history) == 1
        assert game.history[0] is None

    def test_pass_does_not_update_board(self) -> None:
        """Board state unchanged after pass (None) move."""
        game = ClassicOthello()
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = EMPTY
        game.next_color = WHITE
        board_before = game.board.copy()
        game.play_move(None)
        assert np.array_equal(game.board, board_before)

    def test_pass_switches_player(self) -> None:
        """next_color toggles even on pass."""
        game = ClassicOthello()
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = EMPTY
        game.next_color = WHITE
        game.play_move(None)
        assert game.next_color == BLACK


class TestGetAllValidMoves:
    """Test get_all_valid_moves method."""

    def test_returns_valid_moves_for_classic_opening(self, classic_game: ClassicOthello) -> None:
        """BLACK opening moves: {d3, c4, f5, e6}."""
        valid_moves = classic_game.get_all_valid_moves()
        expected_moves = {"d3", "c4", "f5", "e6"}
        assert set(valid_moves) == expected_moves

    def test_caches_result(self, classic_game: ClassicOthello) -> None:
        """Second call returns same object without recomputation."""
        moves1 = classic_game.get_all_valid_moves()
        moves2 = classic_game.get_all_valid_moves()
        # Should be the same cached list
        assert moves1 is moves2

    def test_cache_invalidated_after_move(self, classic_game: ClassicOthello) -> None:
        """After play_move, next call recomputes."""
        moves1 = classic_game.get_all_valid_moves()
        classic_game.play_move("d3")
        moves2 = classic_game.get_all_valid_moves()
        # Should be different objects
        assert moves1 is not moves2
        # And different contents
        assert set(moves1) != set(moves2)

    def test_returns_none_when_no_valid_squares(self) -> None:
        """When player must pass, returns [None]."""
        game = ClassicOthello()
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = EMPTY
        game.next_color = WHITE
        valid_moves = game.get_all_valid_moves()
        assert valid_moves == [None]

    def test_pass_is_only_move_when_all_squares_blocked(self) -> None:
        """Board nearly full, no valid square — returns [None]."""
        game = ClassicOthello()
        # Fill board with alternating pattern where no moves possible
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = WHITE
        game.board[0, 1] = WHITE
        game.next_color = WHITE
        valid_moves = game.get_all_valid_moves()
        assert valid_moves == [None]


class TestGetRandomValidMove:
    """Test get_random_valid_move method."""

    def test_returns_valid_move(self, classic_game: ClassicOthello) -> None:
        """Returned move is in get_all_valid_moves()."""
        valid_moves = classic_game.get_all_valid_moves()
        random_move = classic_game.get_random_valid_move()
        assert random_move in valid_moves

    def test_returns_none_when_no_square_moves(self) -> None:
        """When all squares invalid, returns None (pass)."""
        game = ClassicOthello()
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = EMPTY
        game.next_color = WHITE
        random_move = game.get_random_valid_move()
        assert random_move is None

    def test_randomness(self, classic_game: ClassicOthello) -> None:
        """Multiple calls can return different moves (statistical)."""
        moves_seen = set()
        for _ in range(20):
            game = ClassicOthello()
            move = game.get_random_valid_move()
            moves_seen.add(move)
        # Should see more than one move over 20 trials
        assert len(moves_seen) > 1


# ===== TestGenerateRandomGame =====


class TestGenerateRandomGame:
    """Test generate_random_game method."""

    def test_game_produces_history(self, classic_game: ClassicOthello) -> None:
        """History is non-empty after generation."""
        classic_game.generate_random_game()
        assert len(classic_game.history) > 0

    def test_game_terminates_on_double_pass(self) -> None:
        """Game ends when two consecutive passes occur; done=True."""
        game = ClassicOthello()
        # Set up a situation where passes will happen
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = EMPTY
        game.next_color = WHITE
        game.generate_random_game()
        # Should have two passes at the end
        assert len(game.history) >= 2
        assert game.history[-1] is None
        assert game.history[-2] is None
        assert game.done is True

    def test_game_respects_max_steps(self, classic_game: ClassicOthello) -> None:
        """History length <= MAX_STEPS."""
        classic_game.generate_random_game()
        assert len(classic_game.history) <= MAX_STEPS

    def test_board_history_matches_history_length(self, classic_game: ClassicOthello) -> None:
        """len(board_history) == len(history)."""
        classic_game.generate_random_game()
        assert len(classic_game.board_history) == len(classic_game.history)

    def test_first_move_pass_no_crash(self) -> None:
        """Custom board where first player must pass — should not IndexError (bug A-1)."""
        game = ClassicOthello()
        # Create a board where BLACK must pass on first move
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = EMPTY
        game.board[7, 7] = WHITE
        game.next_color = BLACK
        # This should not crash with IndexError
        game.generate_random_game()
        # Should have at least one move (the pass)
        assert len(game.history) >= 1

    def test_deterministic_with_seed(self) -> None:
        """Same np.random seed produces same game."""
        np.random.seed(42)
        game1 = ClassicOthello()
        game1.generate_random_game()
        history1 = game1.history.copy()

        np.random.seed(42)
        game2 = ClassicOthello()
        game2.generate_random_game()
        history2 = game2.history.copy()

        assert history1 == history2


class TestRecoverFromHistory:
    """Test recover_from_history method."""

    def test_recover_produces_same_board(self) -> None:
        """Play a game, save history, recover on new instance — boards match."""
        game1 = ClassicOthello()
        game1.play_move("d3")
        game1.play_move("c3")
        game1.play_move("b3")
        history = game1.get_history()
        final_board = game1.board.copy()

        game2 = ClassicOthello()
        game2.recover_from_history(history)
        assert np.array_equal(game2.board, final_board)

    def test_recover_produces_same_history(self) -> None:
        """Recovered game's history matches input."""
        game1 = ClassicOthello()
        game1.play_move("d3")
        game1.play_move("c3")
        history = game1.get_history()

        game2 = ClassicOthello()
        game2.recover_from_history(history)
        assert game2.history == history

    def test_recover_empty_history(self, classic_game: ClassicOthello) -> None:
        """Empty history list — game stays at initial state."""
        initial_board = classic_game.board.copy()
        classic_game.recover_from_history([])
        assert np.array_equal(classic_game.board, initial_board)
        assert len(classic_game.history) == 0

    def test_recover_with_pass_moves(self) -> None:
        """History containing None moves recovers correctly."""
        game1 = ClassicOthello()
        game1.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game1.board[0, 0] = EMPTY
        game1.next_color = WHITE
        game1.play_move(None)
        game1.play_move(None)
        history = game1.get_history()

        game2 = ClassicOthello()
        game2.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game2.board[0, 0] = EMPTY
        game2.next_color = WHITE
        game2.recover_from_history(history)
        assert game2.history == history


class TestPrintBoard:
    """Test print_board method."""

    def test_does_not_crash(self, classic_game: ClassicOthello) -> None:
        """print_board() executes without error (capture stdout)."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            classic_game.print_board()
            output = captured_output.getvalue()
            assert len(output) > 0
        finally:
            sys.stdout = sys.__stdout__

    def test_output_contains_pieces(self, classic_game: ClassicOthello) -> None:
        """Output contains 'B' and 'W' characters."""
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            classic_game.print_board()
            output = captured_output.getvalue()
            assert "B" in output
            assert "W" in output
        finally:
            sys.stdout = sys.__stdout__


class TestGetHistory:
    """Test get_history method."""

    def test_returns_copy(self, classic_game: ClassicOthello) -> None:
        """Modifying returned list doesn't affect game.history."""
        classic_game.play_move("d3")
        history = classic_game.get_history()
        history.append("invalid_move")
        assert len(classic_game.history) == 1
        assert classic_game.history[0] == "d3"

    def test_matches_played_moves(self, classic_game: ClassicOthello) -> None:
        """Returned list matches sequence of played moves."""
        moves = ["d3", "c3", "b3"]
        for move in moves:
            classic_game.play_move(move)
        history = classic_game.get_history()
        assert history == moves


class TestGetBoardHistory:
    """Test get_board_history method."""

    def test_returns_copy(self, classic_game: ClassicOthello) -> None:
        """Modifying returned list doesn't affect game.board_history."""
        classic_game.play_move("d3")
        board_history = classic_game.get_board_history()
        board_history.append(np.zeros((BOARD_DIM, BOARD_DIM)))
        assert len(classic_game.board_history) == 1

    def test_board_snapshots_are_independent(self, classic_game: ClassicOthello) -> None:
        """Each board in history is a distinct array (not shared reference)."""
        classic_game.play_move("d3")
        classic_game.play_move("c3")
        board_history = classic_game.get_board_history()
        # Modify first snapshot
        board_history[0][0, 0] = 999
        # Second snapshot should be unaffected
        assert board_history[1][0, 0] != 999
        # Game's board should also be unaffected
        assert classic_game.board[0, 0] != 999


# ===== TestPlotBoard =====


class TestPlotBoard:
    """Test plot_board method (basic functionality)."""

    def test_returns_axes_object(self, classic_game: ClassicOthello) -> None:
        """Returns matplotlib Axes instance."""
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        ax = classic_game.plot_board()
        assert isinstance(ax, Axes)
        plt.close()

    def test_with_provided_axes(self, classic_game: ClassicOthello) -> None:
        """Plots on a pre-created axes."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        returned_ax = classic_game.plot_board(ax=ax)
        assert returned_ax is ax
        plt.close()

    def test_with_shading_array(self, classic_game: ClassicOthello) -> None:
        """Shading with numpy array doesn't crash."""
        import matplotlib.pyplot as plt

        shading = np.random.rand(BOARD_DIM, BOARD_DIM)
        ax = classic_game.plot_board(shading=shading)
        assert ax is not None
        plt.close()

    def test_with_shading_valid_string(self, classic_game: ClassicOthello) -> None:
        """shading='valid' highlights valid moves."""
        import matplotlib.pyplot as plt

        ax = classic_game.plot_board(shading="valid")
        assert ax is not None
        plt.close()

    def test_with_move_highlight(self, classic_game: ClassicOthello) -> None:
        """move=(x,y) renders without error."""
        import matplotlib.pyplot as plt

        ax = classic_game.plot_board(move=(3, 3))
        assert ax is not None
        plt.close()

    def test_with_annotations(self, classic_game: ClassicOthello) -> None:
        """annotate_cells=True renders text on cells."""
        import matplotlib.pyplot as plt

        shading = np.random.rand(BOARD_DIM, BOARD_DIM)
        ax = classic_game.plot_board(shading=shading, annotate_cells=True)
        assert ax is not None
        plt.close()

    def test_shading_shape_mismatch_raises(self, classic_game: ClassicOthello) -> None:
        """Shading with wrong shape raises ValueError (after A-8 fix)."""
        import matplotlib.pyplot as plt

        wrong_shape_shading = np.zeros((4, 4))
        with pytest.raises(ValueError, match="same shape"):
            classic_game.plot_board(shading=wrong_shape_shading)
        plt.close()

    def test_custom_cmap(self, classic_game: ClassicOthello) -> None:
        """cmap='Blues' works correctly."""
        import matplotlib.pyplot as plt

        shading = np.random.rand(BOARD_DIM, BOARD_DIM)
        ax = classic_game.plot_board(shading=shading, cmap="Blues")
        assert ax is not None
        plt.close()
