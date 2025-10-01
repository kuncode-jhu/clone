"""Utility helpers for diphone-based decoding.

This module implements helpers to build diphone targets from monophone
sequences and to marginalise diphone log-probabilities back into the
phoneme space.  The implementation mirrors the description of the
DCoND approach in Li et al. (2024).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

# The order matches the LOGIT_PHONE_DEF list that is already used by the
# baseline codebase.  Importing the original constant would pull in heavy
# optional dependencies (``g2p_en``), so we replicate the definition here to
# keep the utilities self-contained.
PHONEME_LIST: List[str] = [
    "BLANK", "SIL",
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
]


@dataclass
class DiphoneConfig:
    """Light-weight container describing the diphone training setup."""

    blank_id: int = 0
    sil_id: int = 1
    include_terminal_sil: bool = True


class DiphoneCodec:
    """Bidirectional mapping between diphone tuples and integer ids.

    The codec treats diphones as ordered pairs of phoneme ids where the first
    element corresponds to the *preceding* phoneme and the second element to
    the current phoneme.  The blank symbol is excluded from the Cartesian
    product because it is handled directly by the CTC loss.  The resulting
    space contains ``(n_phonemes_without_blank) ** 2`` entries which are stored
    after a dedicated blank logit at index zero.
    """

    def __init__(self, n_phoneme_classes: int, config: DiphoneConfig | None = None):
        if config is None:
            config = DiphoneConfig()
        if n_phoneme_classes <= config.blank_id:
            raise ValueError("The number of phoneme classes must exceed the blank id")

        self.config = config
        self.n_phoneme_classes = n_phoneme_classes
        self.n_effective_phonemes = n_phoneme_classes - 1  # exclude blank
        if self.config.sil_id <= 0 or self.config.sil_id >= n_phoneme_classes:
            raise ValueError("Invalid SIL phoneme id provided to DiphoneCodec")

    @property
    def num_output_classes(self) -> int:
        """Total number of output classes including the CTC blank."""

        return 1 + self.n_effective_phonemes * self.n_effective_phonemes

    def encode(self, prev_id: int, curr_id: int) -> int:
        """Return the class index for the ``prev_id -> curr_id`` diphone."""

        if prev_id == self.config.blank_id or curr_id == self.config.blank_id:
            raise ValueError("Blank symbols are not valid diphone components")
        prev_adj = prev_id - 1
        curr_adj = curr_id - 1
        if not (0 <= prev_adj < self.n_effective_phonemes):
            raise ValueError(f"Invalid previous phoneme id: {prev_id}")
        if not (0 <= curr_adj < self.n_effective_phonemes):
            raise ValueError(f"Invalid current phoneme id: {curr_id}")
        return 1 + prev_adj * self.n_effective_phonemes + curr_adj

    def decode(self, diphone_id: int) -> tuple[int, int]:
        """Return the ``(prev_id, curr_id)`` pair for ``diphone_id``."""

        if diphone_id == 0:
            raise ValueError("Index 0 corresponds to the CTC blank and is not a diphone")
        idx = diphone_id - 1
        prev_adj, curr_adj = divmod(idx, self.n_effective_phonemes)
        return prev_adj + 1, curr_adj + 1

    def encode_sequence(self, phoneme_seq: Sequence[int]) -> torch.Tensor:
        """Convert a phoneme sequence into diphone ids.

        The sequence is augmented with a leading ``SIL`` token to capture the
        start-of-sentence context.  Optionally a trailing ``SIL`` diphone is
        appended to encode the transition back to silence.
        """

        prev = self.config.sil_id
        diphone_ids: List[int] = []
        for phoneme in phoneme_seq:
            if phoneme == self.config.blank_id:
                # CTC targets never contain blanks but we guard against it to
                # remain robust to malformed data.
                continue
            diphone_ids.append(self.encode(prev, phoneme))
            prev = phoneme
        if self.config.include_terminal_sil:
            diphone_ids.append(self.encode(prev, self.config.sil_id))
        if not diphone_ids:
            # Degenerate sequences (e.g. empty phoneme labels) are rare but we
            # provide a sensible default so the CTC loss receives at least one
            # target element.
            diphone_ids.append(self.encode(self.config.sil_id, self.config.sil_id))
        return torch.tensor(diphone_ids, dtype=torch.long)

    def marginalise_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Map diphone log-probabilities back into the phoneme space.

        Args:
            log_probs: ``(batch, time, num_output_classes)`` tensor containing
                log-probabilities over the diphone vocabulary including the
                leading blank entry.
        Returns:
            ``(batch, time, n_phoneme_classes)`` tensor with phoneme
            log-probabilities where the blank distribution is copied over and
            non-blank entries are computed via log-sum-exp marginalisation over
            the preceding phoneme axis.
        """

        if log_probs.shape[-1] != self.num_output_classes:
            raise ValueError(
                "The provided log-probabilities do not match the codec's vocabulary"
            )
        blank = log_probs[..., :1]
        diphone = log_probs[..., 1:]
        batch, time, _ = diphone.shape
        diphone = diphone.view(batch, time, self.n_effective_phonemes, self.n_effective_phonemes)
        current = torch.logsumexp(diphone, dim=2)
        return torch.cat([blank, current], dim=-1)

    def extend_sequence_lengths(self, phoneme_lengths: torch.Tensor) -> torch.Tensor:
        """Map phoneme sequence lengths to the corresponding diphone lengths."""

        if not torch.is_tensor(phoneme_lengths):
            phoneme_lengths = torch.tensor(phoneme_lengths, dtype=torch.long)
        adjustment = 1 if self.config.include_terminal_sil else 0
        return phoneme_lengths + adjustment


__all__ = [
    "PHONEME_LIST",
    "DiphoneCodec",
    "DiphoneConfig",
]
