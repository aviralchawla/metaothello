"""Tests for metaothello.mingpt.tokenizer.Tokenizer."""

import pytest

from metaothello.constants import MAX_STEPS, SQUARES
from metaothello.mingpt.tokenizer import Tokenizer


@pytest.fixture
def tok() -> Tokenizer:
    """Fresh Tokenizer instance."""
    return Tokenizer()


class TestTokenizerInit:
    """Test Tokenizer initialisation."""

    def test_vocab_size(self, tok: Tokenizer) -> None:
        """Vocab = 1 PAD + 64 squares + 1 pass = 66."""
        assert tok.vocab_size == 66

    def test_pad_token_at_index_zero(self, tok: Tokenizer) -> None:
        """PAD token maps to index 0."""
        assert tok.pad_token_id == 0
        assert tok.stoi[Tokenizer.PAD_TOKEN] == 0

    def test_all_squares_in_vocab(self, tok: Tokenizer) -> None:
        """Every board square is present in the vocabulary."""
        for square in SQUARES:
            assert square in tok.stoi

    def test_pass_token_is_last(self, tok: Tokenizer) -> None:
        """None (pass) maps to the final vocab index."""
        assert tok.stoi[None] == tok.vocab_size - 1

    def test_stoi_and_itos_are_inverses(self, tok: Tokenizer) -> None:
        """Stoi and itos are inverse mappings."""
        for token, idx in tok.stoi.items():
            assert tok.itos[idx] == token

    def test_vocab_has_no_duplicates(self, tok: Tokenizer) -> None:
        """All indices in stoi are unique."""
        indices = list(tok.stoi.values())
        assert len(indices) == len(set(indices))


class TestEncodeDecode:
    """Test encode and decode methods."""

    def test_encode_square(self, tok: Tokenizer) -> None:
        """Encoding a valid square returns an integer."""
        result = tok.encode(["e6"])
        assert len(result) == 1
        assert isinstance(result[0], int)

    def test_encode_none(self, tok: Tokenizer) -> None:
        """None (pass move) encodes to the last token index."""
        result = tok.encode([None])
        assert result == [tok.vocab_size - 1]

    def test_encode_pad_token(self, tok: Tokenizer) -> None:
        """PAD token encodes to index 0."""
        result = tok.encode([Tokenizer.PAD_TOKEN])
        assert result == [0]

    def test_encode_decode_roundtrip(self, tok: Tokenizer) -> None:
        """Decode(encode(seq)) == seq for a mixed sequence."""
        seq = ["e6", "f4", None, "d3", Tokenizer.PAD_TOKEN]
        assert tok.decode(tok.encode(seq)) == seq

    def test_encode_full_game_length(self, tok: Tokenizer) -> None:
        """Encoding a MAX_STEPS-length sequence produces MAX_STEPS tokens."""
        seq = [SQUARES[i % len(SQUARES)] for i in range(MAX_STEPS)]
        encoded = tok.encode(seq)
        assert len(encoded) == MAX_STEPS

    def test_all_encoded_values_in_range(self, tok: Tokenizer) -> None:
        """Every token ID is within [0, vocab_size)."""
        seq = [*SQUARES, None, Tokenizer.PAD_TOKEN]
        encoded = tok.encode(seq)
        for token_id in encoded:
            assert 0 <= token_id < tok.vocab_size

    def test_encode_invalid_move_raises(self, tok: Tokenizer) -> None:
        """Encoding an unknown move raises KeyError."""
        with pytest.raises(KeyError):
            tok.encode(["z9"])

    def test_decode_invalid_token_raises(self, tok: Tokenizer) -> None:
        """Decoding an out-of-range token ID raises KeyError."""
        with pytest.raises(KeyError):
            tok.decode([9999])

    def test_encode_empty_sequence(self, tok: Tokenizer) -> None:
        """Empty sequence encodes to empty list."""
        assert tok.encode([]) == []

    def test_decode_empty_sequence(self, tok: Tokenizer) -> None:
        """Empty token list decodes to empty list."""
        assert tok.decode([]) == []


class TestEncodeBatch:
    """Test encode_batch and decode_batch methods."""

    def test_encode_batch_returns_list_of_lists(self, tok: Tokenizer) -> None:
        """encode_batch returns a list of token-ID lists."""
        batch = [["e6", "f4"], ["d3", None]]
        result = tok.encode_batch(batch)
        assert isinstance(result, list)
        assert all(isinstance(seq, list) for seq in result)

    def test_encode_batch_length_preserved(self, tok: Tokenizer) -> None:
        """Batch output has the same number of sequences as input."""
        batch = [["e6"], ["f4", None], ["d3", "c5", "b6"]]
        assert len(tok.encode_batch(batch)) == len(batch)

    def test_encode_batch_consistent_with_encode(self, tok: Tokenizer) -> None:
        """encode_batch(seqs)[i] == encode(seqs[i]) for each i."""
        batch = [["e6", "f4"], ["d3", None, "c5"]]
        batch_result = tok.encode_batch(batch)
        for i, seq in enumerate(batch):
            assert batch_result[i] == tok.encode(seq)

    def test_decode_batch_consistent_with_decode(self, tok: Tokenizer) -> None:
        """decode_batch(token_seqs)[i] == decode(token_seqs[i]) for each i."""
        token_seqs = [[1, 2, 3], [4, 5], [0, 65]]
        batch_result = tok.decode_batch(token_seqs)
        for i, tokens in enumerate(token_seqs):
            assert batch_result[i] == tok.decode(tokens)

    def test_encode_decode_batch_roundtrip(self, tok: Tokenizer) -> None:
        """decode_batch(encode_batch(seqs)) == seqs."""
        batch = [["e6", "f4"], [None, "d3"], [Tokenizer.PAD_TOKEN]]
        assert tok.decode_batch(tok.encode_batch(batch)) == batch

    def test_encode_batch_empty(self, tok: Tokenizer) -> None:
        """Empty batch encodes to empty list."""
        assert tok.encode_batch([]) == []


class TestPadSequence:
    """Test pad_sequence method."""

    def test_pad_shorter_sequence(self, tok: Tokenizer) -> None:
        """Sequence shorter than max_length is padded to max_length."""
        padded = tok.pad_sequence([1, 2, 3], max_length=5)
        assert len(padded) == 5

    def test_pad_values_are_pad_token_id(self, tok: Tokenizer) -> None:
        """Appended values are all equal to pad_token_id (0)."""
        padded = tok.pad_sequence([1, 2], max_length=5)
        assert padded[2:] == [tok.pad_token_id] * 3

    def test_pad_original_tokens_preserved(self, tok: Tokenizer) -> None:
        """Original tokens are unchanged after padding."""
        original = [1, 2, 3]
        padded = tok.pad_sequence(original, max_length=5)
        assert padded[:3] == original

    def test_pad_exact_length_unchanged(self, tok: Tokenizer) -> None:
        """Sequence already at max_length is returned unchanged."""
        tokens = [1, 2, 3]
        assert tok.pad_sequence(tokens, max_length=3) == tokens

    def test_pad_too_long_raises(self, tok: Tokenizer) -> None:
        """ValueError when sequence exceeds max_length and truncate=False."""
        with pytest.raises(ValueError, match="max_length"):
            tok.pad_sequence([1, 2, 3, 4, 5], max_length=3)

    def test_pad_too_long_with_truncate(self, tok: Tokenizer) -> None:
        """truncate=True returns the first max_length tokens."""
        result = tok.pad_sequence([1, 2, 3, 4, 5], max_length=3, truncate=True)
        assert result == [1, 2, 3]

    def test_pad_empty_sequence(self, tok: Tokenizer) -> None:
        """Empty sequence is padded entirely with pad_token_id."""
        padded = tok.pad_sequence([], max_length=3)
        assert padded == [tok.pad_token_id] * 3


class TestPadBatch:
    """Test pad_batch method."""

    def test_auto_max_length(self, tok: Tokenizer) -> None:
        """Without max_length, pads to the longest sequence in the batch."""
        seqs = [[1, 2, 3], [4, 5], [6]]
        result = tok.pad_batch(seqs)
        assert all(len(s) == 3 for s in result)

    def test_fixed_max_length(self, tok: Tokenizer) -> None:
        """Explicit max_length controls output length."""
        seqs = [[1], [2, 3]]
        result = tok.pad_batch(seqs, max_length=5)
        assert all(len(s) == 5 for s in result)

    def test_all_sequences_same_length(self, tok: Tokenizer) -> None:
        """All sequences in the output have the same length."""
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        result = tok.pad_batch(seqs)
        lengths = {len(s) for s in result}
        assert len(lengths) == 1

    def test_original_tokens_preserved_in_batch(self, tok: Tokenizer) -> None:
        """Padding does not alter existing tokens in any sequence."""
        seqs = [[1, 2], [3, 4, 5]]
        result = tok.pad_batch(seqs, max_length=5)
        assert result[0][:2] == [1, 2]
        assert result[1][:3] == [3, 4, 5]

    def test_truncate_in_batch(self, tok: Tokenizer) -> None:
        """truncate=True truncates sequences longer than max_length."""
        seqs = [[1, 2, 3, 4, 5], [6, 7]]
        result = tok.pad_batch(seqs, max_length=3, truncate=True)
        assert all(len(s) == 3 for s in result)

    def test_batch_too_long_raises(self, tok: Tokenizer) -> None:
        """ValueError propagates when a sequence is too long and truncate=False."""
        seqs = [[1, 2, 3, 4, 5], [6, 7]]
        with pytest.raises(ValueError, match="max_length"):
            tok.pad_batch(seqs, max_length=3)
