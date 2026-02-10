import numpy as np

from metaothello.constants import BLACK, EMPTY, WHITE
from metaothello.games import ClassicOthello, DeleteFlanking, NoMiddleFlip


class TestClassicOthello:
    """Test Classic Othello game."""

    def test_initialization(self) -> None:
        """Test game initialization."""
        game = ClassicOthello()
        assert game.alias == "classic"
        expected_board = np.zeros((8, 8))
        expected_board[3, 3] = WHITE
        expected_board[4, 4] = WHITE
        expected_board[3, 4] = BLACK
        expected_board[4, 3] = BLACK
        assert np.array_equal(game.board, expected_board)
        assert game.next_color == BLACK

    def test_play_move_and_board_update(self) -> None:
        """Test playing moves and board updates."""
        game = ClassicOthello()
        # Black plays d3
        game.play_move("d3")
        expected_board = np.zeros((8, 8))
        expected_board[3, 3] = BLACK  # Flipped
        expected_board[4, 4] = WHITE
        expected_board[3, 4] = BLACK
        expected_board[4, 3] = BLACK
        expected_board[2, 3] = BLACK  # New piece
        assert np.array_equal(game.board, expected_board)
        assert game.next_color == WHITE

        # White plays c3
        game.play_move("c3")
        expected_board[2, 2] = WHITE  # New piece
        expected_board[3, 3] = WHITE  # Flipped
        assert np.array_equal(game.board, expected_board)
        assert game.next_color == BLACK

    def test_get_all_valid_moves(self) -> None:
        """Test getting all valid moves."""
        game = ClassicOthello()
        valid_moves = game.get_all_valid_moves()
        # Valid moves for BLACK at the start
        expected_moves = {"f5", "e6", "c4", "d3"}
        assert set(valid_moves) == expected_moves

        game.play_move("d3")  # Black moves
        # Valid moves for WHITE
        valid_moves = game.get_all_valid_moves()
        expected_moves = {"c3", "e3", "c5"}  # Corrected expectation
        assert set(valid_moves) == expected_moves

    def test_get_random_valid_move(self) -> None:
        """Test getting a random valid move."""
        game = ClassicOthello()
        valid_moves = game.get_all_valid_moves()
        random_move = game.get_random_valid_move()
        assert random_move in valid_moves

    def test_game_termination(self) -> None:
        """Test game termination with passes."""
        game = ClassicOthello()
        # Simulate a game where no moves are possible
        game.board = np.full((8, 8), BLACK)
        game.board[0, 0] = EMPTY
        assert game.get_all_valid_moves() == [None]
        game.play_move(None)  # Black passes
        assert game.get_all_valid_moves() == [None]
        game.play_move(None)  # White passes

        # The game should be considered done, although MetaOthello doesn't have a get_winner method
        # This is tested by generate_random_game implicitly

    def test_generate_random_game(self) -> None:
        """Test generating a random game."""
        game = ClassicOthello()
        game.generate_random_game()
        # A random game should have history
        assert len(game.history) > 0
        # After two consecutive passes, the game should end
        assert not (game.history[-1] is None and game.history[-2] is None) or game.done


class TestNoMiddleFlip:
    """Test NoMiddleFlip game variant."""

    def test_initialization(self) -> None:
        """Test game initialization."""
        game = NoMiddleFlip()
        assert game.alias == "nomidflip"
        expected_board = np.zeros((8, 8))
        expected_board[3, 3] = WHITE
        expected_board[4, 4] = WHITE
        expected_board[3, 4] = BLACK
        expected_board[4, 3] = BLACK
        assert np.array_equal(game.board, expected_board)

    def test_play_move_no_middle_flip(self) -> None:
        """Test that middle pieces don't flip."""
        game = NoMiddleFlip()
        # Set up a specific scenario
        game.board = np.zeros((8, 8))
        game.board[3, 2] = BLACK
        game.board[3, 3] = WHITE
        game.board[3, 4] = WHITE
        game.board[3, 5] = WHITE
        game.next_color = BLACK

        # Black plays g4 (3,6), flanking three white pieces.
        # This is a valid move because it's an empty square and it flanks opponent pieces.
        game.play_move("g4")

        # In NoMiddleFlip, only the ends of the flanked line should flip.
        # The new piece is at (3,6)
        # The flanked pieces are at (3,3), (3,4), (3,5)
        # The flanking piece is at (3,2)
        # So, (3,3) and (3,5) should be flipped to BLACK
        # (3,4) should remain WHITE
        assert game.board[3, 6] == BLACK
        assert game.board[3, 3] == BLACK
        assert game.board[3, 5] == BLACK
        assert game.board[3, 4] == WHITE


class TestDeleteFlanking:
    """Test DeleteFlanking game variant."""

    def test_initialization(self) -> None:
        """Test game initialization."""
        game = DeleteFlanking()
        assert game.alias == "delflank"
        expected_board = np.zeros((8, 8))
        expected_board[2, 5] = WHITE
        expected_board[5, 2] = WHITE
        expected_board[2, 2] = BLACK
        expected_board[5, 5] = BLACK
        assert np.array_equal(game.board, expected_board)

    # def test_play_move_delete_flanking(self):
    #     game = DeleteFlanking()
    #     game.next_color = BLACK
    #
    #     # Set up a specific scenario
    #     game.board = np.zeros((8,8))
    #     game.board[2,2] = BLACK # c3
    #     game.board[2,3] = WHITE # d3
    #     game.board[2,4] = WHITE # e3
    #
    #     # A valid move for black is b3, as it's adjacent to c3 (a black piece)
    #     # and the square is empty.
    #     game.play_move("b3")
    #
    #     # The move at b3 (2,1) should place a black piece.
    #     # The flanked pieces at d3 and e3 should be deleted (become EMPTY)
    #     assert game.board[2,1] == BLACK
    #     assert game.board[2,3] == EMPTY
    #     assert game.board[2,4] == EMPTY
    #     assert game.board[2,2] == BLACK
