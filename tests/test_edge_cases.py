"""Cross-cutting edge case tests for MetaOthello."""

import numpy as np

from metaothello.constants import BLACK, BOARD_DIM, EMPTY, WHITE
from metaothello.games import ClassicOthello
from metaothello.rules.validation import is_in_board


class TestBoardBoundaries:
    """Test board boundary handling."""

    def test_all_directions_checked_at_corner_00(self) -> None:
        """Move at (0,0) — only 3 of 8 directions are in-bounds."""
        game = ClassicOthello()
        # Set up a scenario where (0,0) could be valid
        game.board = np.zeros((BOARD_DIM, BOARD_DIM))
        game.board[0, 1] = WHITE
        game.board[0, 2] = BLACK
        game.next_color = BLACK
        # (0,0) should be valid (flanks horizontally)
        assert game.is_valid_move("a1")

    def test_all_directions_checked_at_corner_77(self) -> None:
        """Move at (7,7) — only 3 of 8 directions are in-bounds."""
        game = ClassicOthello()
        game.board = np.zeros((BOARD_DIM, BOARD_DIM))
        game.board[7, 6] = WHITE
        game.board[7, 5] = BLACK
        game.next_color = BLACK
        # (7,7) should be valid (flanks horizontally)
        assert game.is_valid_move("h8")

    def test_all_directions_checked_at_edge_middle(self) -> None:
        """Move at (0,4) — only 5 of 8 directions are in-bounds."""
        game = ClassicOthello()
        game.board = np.zeros((BOARD_DIM, BOARD_DIM))
        game.board[0, 3] = WHITE
        game.board[0, 2] = BLACK
        game.next_color = BLACK
        # (0,4) should be valid (flanks horizontally)
        assert game.is_valid_move("e1")

    def test_is_in_board_boundary_values(self) -> None:
        """is_in_board(0,0)=True, (-1,0)=False, (8,0)=False, (7,7)=True."""
        assert is_in_board(0, 0)
        assert not is_in_board(-1, 0)
        assert not is_in_board(0, -1)
        assert not is_in_board(8, 0)
        assert not is_in_board(0, 8)
        assert is_in_board(7, 7)
        assert is_in_board(3, 4)


class TestPassMechanics:
    """Test pass move mechanics."""

    def test_pass_only_valid_when_no_square_moves(self) -> None:
        """If any square move exists, is_valid_move(None) returns False."""
        game = ClassicOthello()
        # At start, there are valid square moves
        assert not game.is_valid_move(None)

    def test_consecutive_passes_end_game(self) -> None:
        """Two None moves in history triggers game.done in generate_random_game."""
        game = ClassicOthello()
        # Set up board where both players must pass
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = WHITE
        game.next_color = WHITE
        game.generate_random_game()
        # Game should end after two consecutive passes
        assert game.done is True
        assert len(game.history) >= 2
        assert game.history[-1] is None
        assert game.history[-2] is None

    def test_single_pass_does_not_end_game(self) -> None:
        """One pass followed by a valid move — game continues."""
        game = ClassicOthello()
        # Set up a board where WHITE passes but BLACK can move
        game.board = np.zeros((BOARD_DIM, BOARD_DIM))
        game.board[0, 0] = BLACK
        game.board[0, 1] = WHITE
        game.board[0, 2] = WHITE
        game.board[7, 7] = WHITE
        game.next_color = WHITE
        # WHITE should pass (no valid moves for WHITE)
        valid_white = game.get_all_valid_moves()
        assert valid_white == [None]
        game.play_move(None)
        # Now BLACK should have valid moves
        assert game.next_color == BLACK
        valid_moves = game.get_all_valid_moves()
        # BLACK should have at least one valid square move
        assert len(valid_moves) > 0
        assert None not in valid_moves
        # Game should not be done
        assert game.done is False


# ===== TestGameStateConsistency =====


class TestGameStateConsistency:
    """Test game state consistency across operations."""

    def test_board_history_tracks_all_states(self) -> None:
        """board_history[i] reflects state after move i."""
        game = ClassicOthello()
        game.play_move("d3")
        board_after_first_move = game.board.copy()
        game.play_move("c3")
        board_after_second_move = game.board.copy()

        # Check that board_history[0] matches state after first move
        assert np.array_equal(game.board_history[0], board_after_first_move)
        # Check that board_history[1] matches state after second move
        assert np.array_equal(game.board_history[1], board_after_second_move)

    def test_board_history_independent_of_current_board(self) -> None:
        """Modifying game.board doesn't affect stored board_history entries."""
        game = ClassicOthello()
        game.play_move("d3")
        board_snapshot = game.board_history[0].copy()
        # Modify current board
        game.board[0, 0] = 999
        # Stored history should be unchanged
        assert np.array_equal(game.board_history[0], board_snapshot)
        assert game.board_history[0][0, 0] != 999

    def test_next_color_alternates(self) -> None:
        """After every move, color flips (even on pass)."""
        game = ClassicOthello()
        assert game.next_color == BLACK
        game.play_move("d3")
        assert game.next_color == WHITE
        game.play_move("c3")
        assert game.next_color == BLACK

        # Test with pass
        game.board = np.full((BOARD_DIM, BOARD_DIM), BLACK)
        game.board[0, 0] = EMPTY
        game.next_color = WHITE
        game.play_move(None)
        assert game.next_color == BLACK

    def test_history_matches_board_history_length(self) -> None:
        """len(history) == len(board_history) always."""
        game = ClassicOthello()
        assert len(game.history) == len(game.board_history)

        game.play_move("d3")
        assert len(game.history) == len(game.board_history)

        game.play_move("c3")
        assert len(game.history) == len(game.board_history)

        game.play_move("b3")
        assert len(game.history) == len(game.board_history)


# ===== TestFullGameIntegrity =====


class TestFullGameIntegrity:
    """Test full game integrity and consistency."""

    def test_classic_known_game_sequence(self) -> None:
        """Play a known sequence of moves, verify final board against expected state."""
        game = ClassicOthello()

        # Play a known sequence
        game.play_move("d3")  # BLACK
        game.play_move("c3")  # WHITE
        game.play_move("b3")  # BLACK

        # Verify key pieces are in place
        # After these moves, c3 position (2,2) gets flipped by b3
        assert game.board[2, 1] == BLACK  # b3 position
        assert len(game.history) == 3
        # Verify that the game state is consistent (should be WHITE's turn after 3 moves)
        assert game.next_color == WHITE

    def test_recover_and_continue(self) -> None:
        """Recover from partial history, then continue playing — state is consistent."""
        game1 = ClassicOthello()
        game1.play_move("d3")
        game1.play_move("c3")
        history = game1.get_history()
        board_after_two_moves = game1.board.copy()

        # Recover on new game
        game2 = ClassicOthello()
        game2.recover_from_history(history)
        assert np.array_equal(game2.board, board_after_two_moves)

        # Continue playing on both games
        game1.play_move("b3")
        game2.play_move("b3")

        # Boards should still match
        assert np.array_equal(game1.board, game2.board)
        assert game1.history == game2.history

    def test_two_games_same_seed_identical(self) -> None:
        """Same random seed produces identical game history and board state."""
        np.random.seed(123)
        game1 = ClassicOthello()
        game1.generate_random_game()
        history1 = game1.history.copy()
        board1 = game1.board.copy()

        np.random.seed(123)
        game2 = ClassicOthello()
        game2.generate_random_game()
        history2 = game2.history.copy()
        board2 = game2.board.copy()

        assert history1 == history2
        assert np.array_equal(board1, board2)
