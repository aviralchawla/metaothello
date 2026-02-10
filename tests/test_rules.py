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
)


class TestInitializationRules:
    """Test initialization rules."""

    def test_classic_initialization(self) -> None:
        """Test classic initialization."""
        game = MetaOthello(ClassicInitialization, [], [])
        expected_board = np.zeros((8, 8))
        expected_board[3, 3] = WHITE
        expected_board[4, 4] = WHITE
        expected_board[3, 4] = BLACK
        expected_board[4, 3] = BLACK
        assert np.array_equal(game.board, expected_board)

    def test_open_spread_initialization(self) -> None:
        """Test open spread initialization."""
        game = MetaOthello(OpenSpreadInitialization, [], [])
        expected_board = np.zeros((8, 8))
        expected_board[2, 5] = WHITE
        expected_board[5, 2] = WHITE
        expected_board[2, 2] = BLACK
        expected_board[5, 5] = BLACK
        assert np.array_equal(game.board, expected_board)


class TestValidationRules:
    """Test validation rules."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.game = MetaOthello(ClassicInitialization, [], [])

    def test_available_rule(self) -> None:
        """Test available rule."""
        assert AvailableRule.is_valid(self.game, 2, 3)
        assert not AvailableRule.is_valid(self.game, 3, 3)

    def test_standard_flanking_validation_rule(self) -> None:
        """Test standard flanking validation rule."""
        assert StandardFlankingValidationRule.is_valid(self.game, 2, 3)
        assert not StandardFlankingValidationRule.is_valid(self.game, 0, 0)

    def test_neighbor_validation_rule(self) -> None:
        """Test neighbor validation rule."""
        self.game.board[2, 2] = BLACK
        assert NeighborValidationRule.is_valid(self.game, 2, 3)
        assert not NeighborValidationRule.is_valid(self.game, 0, 0)


class TestUpdateRules:
    """Test update rules."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.game = MetaOthello(ClassicInitialization, [], [])

    def test_standard_flanking_update_rule(self) -> None:
        """Test standard flanking update rule."""
        StandardFlankingUpdateRule.update(self.game, 2, 3)
        assert self.game.board[2, 3] == BLACK
        assert self.game.board[3, 3] == BLACK

    def test_delete_flanking_update_rule(self) -> None:
        """Test delete flanking update rule."""
        DeleteFlankingUpdateRule.update(self.game, 2, 3)
        assert self.game.board[2, 3] == BLACK
        assert self.game.board[3, 3] == EMPTY

    def test_no_middle_flip_update_rule(self) -> None:
        """Test no middle flip update rule."""
        self.game.board[1, 3] = WHITE
        NoMiddleFlipUpdateRule.update(self.game, 0, 3)
        assert self.game.board[0, 3] == BLACK
        assert self.game.board[1, 3] == WHITE
