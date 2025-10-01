"""Utility helpers for the DCoND phoneme+diphone decoding pipeline."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import torch

# Default phoneme vocabulary used by the Brain-to-Text baseline. The index of
# each entry matches the numerical class id stored in the dataset.
PHONEME_VOCAB: Tuple[str, ...] = (
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',  # Silence token used in the public dataset.
)

SILENCE_SYMBOL = ' | '
DEFAULT_BLANK_ID = 0
DEFAULT_SILENCE_ID = PHONEME_VOCAB.index(SILENCE_SYMBOL)


def get_phoneme_vocab_size() -> int:
    """Return the size of the canonical phoneme vocabulary (including blank)."""

    return len(PHONEME_VOCAB)


def diphone_vocab_size(phoneme_vocab_size: int) -> int:
    """Return the number of diphone classes (including the blank class)."""

    # Exclude the CTC blank. Every remaining phoneme (including SIL) can pair
    # with every other phoneme.
    real_phonemes = phoneme_vocab_size - 1
    return 1 + real_phonemes * real_phonemes


def _encode_diphone(prev_phone: int, curr_phone: int, phoneme_vocab_size: int) -> int:
    """Encode a (previous, current) phoneme pair into a single diphone id."""

    if prev_phone == DEFAULT_BLANK_ID or curr_phone == DEFAULT_BLANK_ID:
        raise ValueError('CTC blank should not be part of diphone subclasses.')

    real_phonemes = phoneme_vocab_size - 1
    prev_index = prev_phone - 1
    curr_index = curr_phone - 1
    if prev_index < 0 or curr_index < 0:
        raise ValueError('Phoneme ids must be >= 1 once the blank is removed.')
    return 1 + prev_index * real_phonemes + curr_index


def phoneme_to_diphone_sequence(
    phoneme_sequence: torch.Tensor,
    *,
    phoneme_vocab_size: int = get_phoneme_vocab_size(),
    silence_id: int = DEFAULT_SILENCE_ID,
    blank_id: int = DEFAULT_BLANK_ID,
    hold_strategy: str = 'self_loop',
) -> torch.Tensor:
    """Convert a phoneme label sequence into diphone subclasses.

    Parameters
    ----------
    phoneme_sequence:
        1D tensor containing the phoneme ids for a single utterance (without
        padding). The tensor may contain silence tokens but should not contain
        the CTC blank.
    phoneme_vocab_size:
        Total number of phoneme classes (including the blank token).
    silence_id:
        Numerical id corresponding to the SIL token.
    blank_id:
        Numerical id corresponding to the CTC blank token.
    hold_strategy:
        Strategy that defines how self transitions are inserted. Currently the
        only supported value is ``"self_loop"`` which inserts a ``p -> p``
        self-transition for every phoneme. The argument is provided to make it
        easy to experiment with alternative strategies without touching the
        dataset loader.

    Returns
    -------
    torch.Tensor
        Tensor containing the encoded diphone ids.
    """

    if hold_strategy != 'self_loop':
        raise ValueError(
            "Unsupported hold_strategy '%s'. Only 'self_loop' is currently "
            'implemented.' % hold_strategy
        )

    if phoneme_sequence.ndim != 1:
        raise ValueError('phoneme_sequence must be a 1D tensor.')

    seq = [int(p) for p in phoneme_sequence.tolist() if p != blank_id]
    if not seq:
        # If for some reason the sequence is empty we still emit the boundary
        # transitions to keep the training target well defined.
        seq = [silence_id]

    encoded: List[int] = []
    prev = silence_id
    for phoneme in seq:
        # Skip blanks/zeros that might be left in the array.
        if phoneme == blank_id:
            continue
        encoded.append(_encode_diphone(prev, phoneme, phoneme_vocab_size))
        encoded.append(_encode_diphone(phoneme, phoneme, phoneme_vocab_size))
        prev = phoneme

    encoded.append(_encode_diphone(prev, silence_id, phoneme_vocab_size))
    return torch.tensor(encoded, dtype=torch.long)


@lru_cache(maxsize=None)
def _diphone_component_tensors(
    phoneme_vocab_size: int,
    blank_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return tensors describing the mapping between diphones and phonemes."""

    components: List[Tuple[int, int]] = [(blank_id, blank_id)]
    phoneme_ids = [p for p in range(phoneme_vocab_size) if p != blank_id]
    for prev in phoneme_ids:
        for curr in phoneme_ids:
            components.append((prev, curr))

    pairs = torch.tensor(components, dtype=torch.long)
    return pairs[:, 0], pairs[:, 1]


def marginalise_diphone_log_probs(
    log_probs: torch.Tensor,
    *,
    phoneme_vocab_size: int = get_phoneme_vocab_size(),
    blank_id: int = DEFAULT_BLANK_ID,
) -> torch.Tensor:
    """Marginalise diphone log-probabilities into single phoneme log-probs."""

    if log_probs.ndim != 3:
        raise ValueError('Expected diphone log-probabilities of shape [B, T, V].')

    device = log_probs.device
    prev_components, curr_components = _diphone_component_tensors(
        phoneme_vocab_size, blank_id
    )
    prev_components = prev_components.to(device)
    curr_components = curr_components.to(device)

    batch, time, _ = log_probs.shape
    phoneme_log_probs = log_probs.new_full(
        (batch, time, phoneme_vocab_size), float('-inf')
    )

    # The diphone blank is aligned with the phoneme blank. By convention it is
    # stored at index 0 in both vocabularies.
    phoneme_log_probs[..., blank_id] = log_probs[..., 0]

    for phoneme_id in range(phoneme_vocab_size):
        if phoneme_id == blank_id:
            continue
        mask = curr_components == phoneme_id
        if not torch.any(mask):
            continue
        selected = log_probs[..., mask]
        phoneme_log_probs[..., phoneme_id] = torch.logsumexp(selected, dim=-1)

    return phoneme_log_probs


def current_diphone_component_mask(
    phoneme_vocab_size: int = get_phoneme_vocab_size(),
    blank_id: int = DEFAULT_BLANK_ID,
) -> torch.Tensor:
    """Return the mapping from diphone ids to the current phoneme ids."""

    _, curr = _diphone_component_tensors(phoneme_vocab_size, blank_id)
    return curr
