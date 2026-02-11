"""Visualization tests for plot_board functionality.

These tests are marked as slow since they involve matplotlib rendering.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from metaothello.constants import BOARD_DIM
from metaothello.games import ClassicOthello


@pytest.mark.slow
class TestPlotBoard:
    """Test plot_board visualization functionality."""

    def test_basic_plot_returns_axes(self, classic_game: ClassicOthello) -> None:
        """ClassicOthello().plot_board() returns Axes."""
        ax = classic_game.plot_board()
        assert isinstance(ax, Axes)
        plt.close()

    def test_plot_with_existing_axes(self, classic_game: ClassicOthello) -> None:
        """Pass pre-created axes — same axes returned."""
        _, ax = plt.subplots()
        returned_ax = classic_game.plot_board(ax=ax)
        assert returned_ax is ax
        # Verify something was drawn on the axes (check artists or patches)
        assert len(ax.artists) > 0 or len(ax.patches) > 0
        plt.close()

    def test_plot_with_valid_shading(self, classic_game: ClassicOthello) -> None:
        """shading='valid' works without error."""
        ax = classic_game.plot_board(shading="valid")
        assert isinstance(ax, Axes)
        # Valid moves should be highlighted
        # We can't easily verify the exact rendering, but we can check it doesn't crash
        plt.close()

    def test_plot_with_numpy_shading(self, classic_game: ClassicOthello) -> None:
        """shading=np.ones((8,8)) renders."""
        shading = np.ones((BOARD_DIM, BOARD_DIM))
        ax = classic_game.plot_board(shading=shading)
        assert isinstance(ax, Axes)
        # Should have rectangles for shading (check artists or patches)
        assert len(ax.artists) > 0 or len(ax.patches) > 0
        plt.close()

    def test_plot_with_nan_shading(self, classic_game: ClassicOthello) -> None:
        """Shading with NaN values renders (annotations skip NaN)."""
        shading = np.ones((BOARD_DIM, BOARD_DIM))
        shading[0, 0] = np.nan
        shading[3, 3] = np.nan
        ax = classic_game.plot_board(shading=shading, annotate_cells=True)
        assert isinstance(ax, Axes)
        # Should render without error even with NaN values
        plt.close()

    def test_plot_with_move_highlight(self, classic_game: ClassicOthello) -> None:
        """move=(3,3) highlights cell."""
        ax = classic_game.plot_board(move=(3, 3))
        assert isinstance(ax, Axes)
        # Should have a highlight rectangle (check artists or patches)
        assert len(ax.artists) > 0 or len(ax.patches) > 0
        plt.close()

    def test_plot_with_annotations(self, classic_game: ClassicOthello) -> None:
        """annotate_cells=True adds text."""
        shading = np.random.rand(BOARD_DIM, BOARD_DIM)
        ax = classic_game.plot_board(shading=shading, annotate_cells=True)
        assert isinstance(ax, Axes)
        # Should have text annotations
        assert len(ax.texts) > 0
        plt.close()

    def test_plot_shape_mismatch_raises(self, classic_game: ClassicOthello) -> None:
        """shading=np.zeros((4,4)) raises error."""
        wrong_shape_shading = np.zeros((4, 4))
        with pytest.raises(ValueError, match="same shape"):
            classic_game.plot_board(shading=wrong_shape_shading)
        plt.close()

    def test_plot_custom_cmap_and_vmin_vmax(self, classic_game: ClassicOthello) -> None:
        """Custom colormap and value range accepted."""
        shading = np.random.rand(BOARD_DIM, BOARD_DIM) * 100
        ax = classic_game.plot_board(
            shading=shading,
            cmap="Blues",
            vmin=0,
            vmax=100,
            annotate_cells=True,
        )
        assert isinstance(ax, Axes)
        # Should render with custom colormap (check artists, patches, or texts)
        assert len(ax.artists) > 0 or len(ax.patches) > 0 or len(ax.texts) > 0
        plt.close()
