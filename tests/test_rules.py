"""Comprehensive tests for all rule implementations."""

import numpy as np

from metaothello.constants import BLACK, EMPTY, WHITE
from metaothello.metaothello import MetaOthello
from metaothello.rules.initialization import ClassicInitialization, OpenSpreadInitialization
from metaothello.rules.update import (
    DeleteFlankingUpdateRule,
    NoMiddleFlipUpdateRule,
    StandardFlankingUpdateRule,
)
from metaothello.rules.validation import (
    AvailableRule,
    NeighborValidationRule,
    StandardFlankingValidationRule,
    is_in_board,
)


class TestClassicInitialization:
    """Test classic initialization rule."""

    def test_board_shape(self) -> None:
        """Board is 8x8 numpy array."""
        game = MetaOthello(ClassicInitialization, [], [])
        assert game.board.shape == (8, 8)

    def test_piece_placement(self) -> None:
        """WHITE at (3,3) and (4,4), BLACK at (3,4) and (4,3)."""
        game = MetaOthello(ClassicInitialization, [], [])
        assert game.board[3, 3] == WHITE
        assert game.board[4, 4] == WHITE
        assert game.board[3, 4] == BLACK
        assert game.board[4, 3] == BLACK

    def test_empty_squares(self) -> None:
        """All other 60 squares are EMPTY."""
        game = MetaOthello(ClassicInitialization, [], [])
        # Count pieces
        black_count = np.sum(game.board == BLACK)
        white_count = np.sum(game.board == WHITE)
        empty_count = np.sum(game.board == EMPTY)
        assert black_count == 2
        assert white_count == 2
        assert empty_count == 60

    def test_piece_count(self) -> None:
        """Exactly 2 WHITE and 2 BLACK pieces."""
        game = MetaOthello(ClassicInitialization, [], [])
        assert np.sum(game.board == BLACK) == 2
        assert np.sum(game.board == WHITE) == 2

    def test_reinitialize_resets_board(self) -> None:
        """Modify board, re-call init_board, verify reset to starting position."""
        game = MetaOthello(ClassicInitialization, [], [])
        # Modify the board
        game.board[0, 0] = BLACK
        game.board[7, 7] = WHITE
        # Reinitialize
        ClassicInitialization.init_board(game)
        # Verify it's back to the starting position
        expected_board = np.zeros((8, 8))
        expected_board[3, 3] = WHITE
        expected_board[4, 4] = WHITE
        expected_board[3, 4] = BLACK
        expected_board[4, 3] = BLACK
        assert np.array_equal(game.board, expected_board)


class TestOpenSpreadInitialization:
    """Test open spread initialization rule."""

    def test_board_shape(self) -> None:
        """Board is 8x8 numpy array."""
        game = MetaOthello(OpenSpreadInitialization, [], [])
        assert game.board.shape == (8, 8)

    def test_piece_placement(self) -> None:
        """WHITE at (2,5) and (5,2), BLACK at (2,2) and (5,5)."""
        game = MetaOthello(OpenSpreadInitialization, [], [])
        assert game.board[2, 5] == WHITE
        assert game.board[5, 2] == WHITE
        assert game.board[2, 2] == BLACK
        assert game.board[5, 5] == BLACK

    def test_empty_squares(self) -> None:
        """All other 60 squares are EMPTY."""
        game = MetaOthello(OpenSpreadInitialization, [], [])
        black_count = np.sum(game.board == BLACK)
        white_count = np.sum(game.board == WHITE)
        empty_count = np.sum(game.board == EMPTY)
        assert black_count == 2
        assert white_count == 2
        assert empty_count == 60

    def test_piece_count(self) -> None:
        """Exactly 2 WHITE and 2 BLACK pieces."""
        game = MetaOthello(OpenSpreadInitialization, [], [])
        assert np.sum(game.board == BLACK) == 2
        assert np.sum(game.board == WHITE) == 2


# ===== Validation Rules =====


class TestAvailableRule:
    """Test AvailableRule validation."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.game = MetaOthello(ClassicInitialization, [], [])

    def test_empty_square_is_valid(self) -> None:
        """Empty square returns True."""
        assert AvailableRule.is_valid(self.game, 0, 0)

    def test_black_occupied_invalid(self) -> None:
        """Square with BLACK piece returns False."""
        assert not AvailableRule.is_valid(self.game, 3, 4)  # BLACK at (3,4)

    def test_white_occupied_invalid(self) -> None:
        """Square with WHITE piece returns False."""
        assert not AvailableRule.is_valid(self.game, 3, 3)  # WHITE at (3,3)

    def test_all_empty_squares_valid_on_empty_board(self) -> None:
        """Every square on an empty board returns True."""
        self.game.board = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                assert AvailableRule.is_valid(self.game, i, j)


class TestStandardFlankingValidationRule:
    """Test StandardFlankingValidationRule."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.game = MetaOthello(ClassicInitialization, [], [])

    def test_valid_flanking_move_horizontal(self) -> None:
        """Classic opening: (2,3) is valid for BLACK (flanks WHITE at (3,3))."""
        assert StandardFlankingValidationRule.is_valid(self.game, 2, 3)

    def test_valid_flanking_move_vertical(self) -> None:
        """Classic opening: verify vertical flanking."""
        # (3,2) should flank vertically for BLACK
        assert StandardFlankingValidationRule.is_valid(self.game, 3, 2)

    def test_valid_flanking_move_diagonal(self) -> None:
        """Set up diagonal flanking scenario."""
        self.game.board = np.zeros((8, 8))
        self.game.board[2, 2] = BLACK
        self.game.board[3, 3] = WHITE
        self.game.board[5, 5] = BLACK
        self.game.next_color = BLACK
        # (4,4) should flank diagonally from (2,2) through (3,3) to (4,4)
        assert StandardFlankingValidationRule.is_valid(self.game, 4, 4)

    def test_no_flanking_returns_false(self) -> None:
        """Move with no opponent neighbors."""
        # (0,0) has no adjacent opponent pieces
        assert not StandardFlankingValidationRule.is_valid(self.game, 0, 0)

    def test_adjacent_to_own_color_without_flanking(self) -> None:
        """Adjacent to own piece but doesn't flank — should be False."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 3] = BLACK
        self.game.board[3, 4] = BLACK
        self.game.next_color = BLACK
        # (3,5) is adjacent to own piece but doesn't flank
        assert not StandardFlankingValidationRule.is_valid(self.game, 3, 5)

    def test_flanking_at_board_edge(self) -> None:
        """Flanking sequence along row 0 or column 0."""
        self.game.board = np.zeros((8, 8))
        self.game.board[0, 0] = BLACK
        self.game.board[0, 1] = WHITE
        self.game.board[0, 2] = BLACK
        self.game.next_color = BLACK
        # (0,3) should not flank (no closing piece)
        assert not StandardFlankingValidationRule.is_valid(self.game, 0, 3)
        # But (0,0) extended: Let's set (0,3) to have a proper flank
        self.game.board[0, 0] = EMPTY
        self.game.board[0, 3] = BLACK
        # Now (0,0) should flank
        assert StandardFlankingValidationRule.is_valid(self.game, 0, 0)

    def test_flanking_at_board_corner(self) -> None:
        """Move at (0,0) or (7,7) with valid flanking."""
        self.game.board = np.zeros((8, 8))
        self.game.board[0, 1] = WHITE
        self.game.board[0, 2] = BLACK
        self.game.next_color = BLACK
        # (0,0) should flank horizontally
        assert StandardFlankingValidationRule.is_valid(self.game, 0, 0)

    def test_gap_in_flanking_sequence(self) -> None:
        """Opponent pieces with EMPTY gap — should be False."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 0] = BLACK
        self.game.board[3, 1] = WHITE
        self.game.board[3, 2] = EMPTY  # Gap
        self.game.board[3, 3] = WHITE
        self.game.board[3, 4] = BLACK
        self.game.next_color = BLACK
        # (3,5) should not flank (gap in sequence)
        # Actually, let's test from (3,5) backwards - no, let's be clearer
        # (3,5) trying to flank towards (3,0) won't work because of gap
        # Let me set this up differently
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 3] = BLACK
        self.game.board[3, 4] = WHITE
        self.game.board[3, 5] = EMPTY  # Gap
        self.game.board[3, 6] = WHITE
        self.game.next_color = BLACK
        # (3,7) can't flank because there's a gap
        assert not StandardFlankingValidationRule.is_valid(self.game, 3, 7)

    def test_multiple_directions_flanked(self) -> None:
        """Move that flanks in 2+ directions (any one direction suffices)."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 3] = BLACK
        # Horizontal flank
        self.game.board[3, 4] = WHITE
        self.game.board[3, 5] = BLACK
        # Vertical flank
        self.game.board[4, 3] = WHITE
        self.game.board[5, 3] = BLACK
        self.game.next_color = BLACK
        # (2,3) should flank vertically (not horizontally, but that's ok)
        # Wait, let me reconsider. Let's make (4,4) the target
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 4] = BLACK  # North
        self.game.board[4, 3] = WHITE  # West neighbor
        self.game.board[4, 5] = WHITE  # East neighbor
        self.game.board[5, 4] = WHITE  # South neighbor
        self.game.board[4, 2] = BLACK  # West closing
        self.game.board[4, 6] = BLACK  # East closing
        self.game.next_color = BLACK
        # (4,4) should flank in both horizontal directions
        assert StandardFlankingValidationRule.is_valid(self.game, 4, 4)


class TestNeighborValidationRule:
    """Test NeighborValidationRule."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.game = MetaOthello(ClassicInitialization, [], [])

    def test_adjacent_to_own_color_valid(self) -> None:
        """Square adjacent to same-color piece returns True."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 3] = BLACK
        self.game.next_color = BLACK
        # (3,4) is adjacent horizontally
        assert NeighborValidationRule.is_valid(self.game, 3, 4)

    def test_adjacent_to_opponent_only_invalid(self) -> None:
        """Only adjacent to opponent pieces returns False."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 3] = WHITE
        self.game.next_color = BLACK
        # (3,4) is adjacent to WHITE only
        assert not NeighborValidationRule.is_valid(self.game, 3, 4)

    def test_no_neighbors_invalid(self) -> None:
        """Isolated empty square returns False."""
        self.game.board = np.zeros((8, 8))
        self.game.next_color = BLACK
        # (0,0) has no neighbors
        assert not NeighborValidationRule.is_valid(self.game, 0, 0)

    def test_diagonal_adjacency(self) -> None:
        """Diagonal neighbor of same color counts."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 3] = BLACK
        self.game.next_color = BLACK
        # (4,4) is diagonally adjacent
        assert NeighborValidationRule.is_valid(self.game, 4, 4)

    def test_at_board_corner(self) -> None:
        """Corner square with limited neighbors."""
        self.game.board = np.zeros((8, 8))
        self.game.board[0, 1] = BLACK
        self.game.next_color = BLACK
        # (0,0) is adjacent to (0,1)
        assert NeighborValidationRule.is_valid(self.game, 0, 0)


class TestIsInBoard:
    """Test is_in_board utility function."""

    def test_boundary_values(self) -> None:
        """is_in_board(0,0)=True, (-1,0)=False, (8,0)=False, (7,7)=True."""
        assert is_in_board(0, 0)
        assert not is_in_board(-1, 0)
        assert not is_in_board(0, -1)
        assert not is_in_board(8, 0)
        assert not is_in_board(0, 8)
        assert is_in_board(7, 7)


# ===== Update Rules =====


class TestStandardFlankingUpdateRule:
    """Test StandardFlankingUpdateRule."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.game = MetaOthello(ClassicInitialization, [], [])

    def test_piece_placed_at_move_location(self) -> None:
        """Board[x,y] == curr_color after update."""
        StandardFlankingUpdateRule.update(self.game, 2, 3)
        assert self.game.board[2, 3] == BLACK

    def test_single_piece_flipped(self) -> None:
        """One opponent piece between new piece and existing own piece flips."""
        StandardFlankingUpdateRule.update(self.game, 2, 3)
        # (3,3) should be flipped from WHITE to BLACK
        assert self.game.board[3, 3] == BLACK

    def test_multiple_pieces_flipped_in_line(self) -> None:
        """3 opponent pieces in a row all flip."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 0] = BLACK
        self.game.board[3, 1] = WHITE
        self.game.board[3, 2] = WHITE
        self.game.board[3, 3] = WHITE
        self.game.board[3, 4] = BLACK
        self.game.next_color = BLACK
        StandardFlankingUpdateRule.update(self.game, 3, 5)
        # All three WHITE pieces should be flipped
        # Wait, (3,5) should flank towards (3,4) and beyond
        # Actually (3,4) is BLACK, so (3,5) can't flank in that direction
        # Let me reconsider
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 0] = BLACK
        self.game.board[3, 1] = WHITE
        self.game.board[3, 2] = WHITE
        self.game.board[3, 3] = WHITE
        self.game.next_color = BLACK
        StandardFlankingUpdateRule.update(self.game, 3, 4)
        # Pieces at (3,1), (3,2), (3,3) should all be flipped to BLACK
        assert self.game.board[3, 1] == BLACK
        assert self.game.board[3, 2] == BLACK
        assert self.game.board[3, 3] == BLACK

    def test_flip_in_multiple_directions(self) -> None:
        """Move that causes flips in 2+ directions simultaneously."""
        self.game.board = np.zeros((8, 8))
        self.game.board[2, 4] = BLACK  # North
        self.game.board[3, 4] = WHITE
        self.game.board[4, 3] = WHITE  # West
        self.game.board[4, 2] = BLACK
        self.game.next_color = BLACK
        StandardFlankingUpdateRule.update(self.game, 4, 4)
        # Both flanked pieces should flip
        assert self.game.board[3, 4] == BLACK  # Flipped vertically
        assert self.game.board[4, 3] == BLACK  # Flipped horizontally

    def test_no_flip_when_no_flanking(self) -> None:
        """Piece placed but no flips occur."""
        self.game.board = np.zeros((8, 8))
        self.game.next_color = BLACK
        StandardFlankingUpdateRule.update(self.game, 0, 0)
        # Piece placed
        assert self.game.board[0, 0] == BLACK
        # No other pieces should have changed
        assert np.sum(self.game.board == BLACK) == 1

    def test_flip_at_board_edge(self) -> None:
        """Flanking along edge row/column."""
        self.game.board = np.zeros((8, 8))
        self.game.board[0, 0] = BLACK
        self.game.board[0, 1] = WHITE
        self.game.next_color = BLACK
        StandardFlankingUpdateRule.update(self.game, 0, 2)
        # (0,1) should be flipped
        assert self.game.board[0, 1] == BLACK

    def test_does_not_flip_beyond_own_piece(self) -> None:
        """Opponent pieces beyond the closing own piece remain unchanged."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 0] = BLACK
        self.game.board[3, 1] = WHITE
        self.game.board[3, 2] = WHITE
        self.game.board[3, 3] = BLACK  # Closing piece
        self.game.board[3, 4] = WHITE  # Beyond closing
        self.game.next_color = BLACK
        StandardFlankingUpdateRule.update(self.game, 3, 5)
        # Pieces up to (3,3) flanked by (3,5)? No, (3,5) is next to (3,4)=WHITE, then (3,3)=BLACK
        # So it should flank (3,4)
        assert self.game.board[3, 4] == BLACK
        # But pieces before (3,3) should not change
        assert self.game.board[3, 1] == WHITE
        assert self.game.board[3, 2] == WHITE


class TestDeleteFlankingUpdateRule:
    """Test DeleteFlankingUpdateRule."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.game = MetaOthello(ClassicInitialization, [], [])

    def test_piece_placed_at_move_location(self) -> None:
        """Board[x,y] == curr_color after update."""
        DeleteFlankingUpdateRule.update(self.game, 2, 3)
        assert self.game.board[2, 3] == BLACK

    def test_single_piece_deleted(self) -> None:
        """One flanked opponent piece becomes EMPTY."""
        DeleteFlankingUpdateRule.update(self.game, 2, 3)
        # (3,3) should be deleted (set to EMPTY)
        assert self.game.board[3, 3] == EMPTY

    def test_multiple_pieces_deleted_in_line(self) -> None:
        """3 flanked opponent pieces all become EMPTY."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 0] = BLACK
        self.game.board[3, 1] = WHITE
        self.game.board[3, 2] = WHITE
        self.game.board[3, 3] = WHITE
        self.game.next_color = BLACK
        DeleteFlankingUpdateRule.update(self.game, 3, 4)
        # All three WHITE pieces should be deleted
        assert self.game.board[3, 1] == EMPTY
        assert self.game.board[3, 2] == EMPTY
        assert self.game.board[3, 3] == EMPTY

    def test_delete_in_multiple_directions(self) -> None:
        """Flanked pieces deleted in 2+ directions."""
        self.game.board = np.zeros((8, 8))
        self.game.board[2, 4] = BLACK  # North
        self.game.board[3, 4] = WHITE
        self.game.board[4, 3] = WHITE  # West
        self.game.board[4, 2] = BLACK
        self.game.next_color = BLACK
        DeleteFlankingUpdateRule.update(self.game, 4, 4)
        # Both flanked pieces should be deleted
        assert self.game.board[3, 4] == EMPTY
        assert self.game.board[4, 3] == EMPTY

    def test_own_pieces_untouched(self) -> None:
        """Own pieces in the vicinity remain unchanged."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 0] = BLACK
        self.game.board[3, 1] = BLACK
        self.game.board[3, 2] = WHITE
        self.game.next_color = BLACK
        DeleteFlankingUpdateRule.update(self.game, 3, 3)
        # Own BLACK pieces should remain
        assert self.game.board[3, 0] == BLACK
        assert self.game.board[3, 1] == BLACK


class TestNoMiddleFlipUpdateRule:
    """Test NoMiddleFlipUpdateRule."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.game = MetaOthello(ClassicInitialization, [], [])

    def test_piece_placed_at_move_location(self) -> None:
        """Board[x,y] == curr_color after update."""
        NoMiddleFlipUpdateRule.update(self.game, 2, 3)
        assert self.game.board[2, 3] == BLACK

    def test_single_piece_flanked_flips(self) -> None:
        """Only 1 piece in sequence — it flips (it's both first and last)."""
        NoMiddleFlipUpdateRule.update(self.game, 2, 3)
        # (3,3) is the only piece in the sequence, so it flips
        assert self.game.board[3, 3] == BLACK

    def test_two_pieces_flanked_both_flip(self) -> None:
        """2 pieces in sequence — both flip (first == index 0, last == index 1)."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 0] = BLACK
        self.game.board[3, 1] = WHITE
        self.game.board[3, 2] = WHITE
        self.game.next_color = BLACK
        NoMiddleFlipUpdateRule.update(self.game, 3, 3)
        # Both WHITE pieces should flip
        assert self.game.board[3, 1] == BLACK
        assert self.game.board[3, 2] == BLACK

    def test_three_pieces_flanked_middle_unchanged(self) -> None:
        """3 pieces: first and last flip, middle stays opponent color."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 0] = BLACK
        self.game.board[3, 1] = WHITE
        self.game.board[3, 2] = WHITE
        self.game.board[3, 3] = WHITE
        self.game.next_color = BLACK
        NoMiddleFlipUpdateRule.update(self.game, 3, 4)
        # First and last should flip, middle stays
        assert self.game.board[3, 1] == BLACK  # First
        assert self.game.board[3, 2] == WHITE  # Middle - unchanged
        assert self.game.board[3, 3] == BLACK  # Last

    def test_five_pieces_flanked_only_endpoints_flip(self) -> None:
        """5 pieces: only positions 0 and 4 flip, positions 1-3 unchanged."""
        self.game.board = np.zeros((8, 8))
        self.game.board[3, 0] = BLACK
        self.game.board[3, 1] = WHITE
        self.game.board[3, 2] = WHITE
        self.game.board[3, 3] = WHITE
        self.game.board[3, 4] = WHITE
        self.game.board[3, 5] = WHITE
        self.game.next_color = BLACK
        NoMiddleFlipUpdateRule.update(self.game, 3, 6)
        # Only first and last flip
        assert self.game.board[3, 1] == BLACK  # First
        assert self.game.board[3, 2] == WHITE  # Middle - unchanged
        assert self.game.board[3, 3] == WHITE  # Middle - unchanged
        assert self.game.board[3, 4] == WHITE  # Middle - unchanged
        assert self.game.board[3, 5] == BLACK  # Last

    def test_multi_direction_only_endpoints_per_direction(self) -> None:
        """Multiple flanking directions each apply endpoint-only logic independently."""
        self.game.board = np.zeros((8, 8))
        # Horizontal: 3 pieces
        self.game.board[4, 0] = BLACK
        self.game.board[4, 1] = WHITE
        self.game.board[4, 2] = WHITE
        self.game.board[4, 3] = WHITE
        # Vertical: 2 pieces
        self.game.board[2, 4] = BLACK
        self.game.board[3, 4] = WHITE
        self.game.next_color = BLACK
        NoMiddleFlipUpdateRule.update(self.game, 4, 4)
        # Horizontal: first and last flip, middle unchanged
        assert self.game.board[4, 1] == BLACK
        assert self.game.board[4, 2] == WHITE  # Middle unchanged
        assert self.game.board[4, 3] == BLACK
        # Vertical: both flip
        assert self.game.board[3, 4] == BLACK
