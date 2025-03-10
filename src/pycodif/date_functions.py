import math
from datetime import date, datetime, timedelta

from pycodif.constants import CODIF_BASE_YEAR


def calc_epoch_base(reference_epoch: int) -> date:
    """Calculates the CODIF epoch based on the reference epoch number.

    These are calculated with 0 being 2000-01-01, and referring to six-month intervals thereafter.

    Using logic from CODIFIO C++ implementation by A. Deller.
    """
    return datetime(
        CODIF_BASE_YEAR + reference_epoch // 2,
        (reference_epoch % 2) * 6 + 1,
        1,
        0,
        0,
        0,
    )


def calc_start_alignment_period_timestamp(
    reference_epoch_date: date, epoch_offset_seconds: int
) -> datetime:
    return reference_epoch_date + timedelta(seconds=epoch_offset_seconds)


def calc_complete_sample_blocks_to_end_of_frame(
    data_frame_number: int, data_array_length: int, sample_block_length: int
) -> int:
    """Used for calculating the timestamp of specific observations

    Calculation is sourced from the CSIRO CODIF specification document.
    Units of data_array_length and sample_block_length are the same, so exact units do not matter here.
    """
    return math.floor((data_frame_number + 1) * data_array_length / sample_block_length)


def calc_number_channel_blocks_per_sample_block(
    sample_block_length_64_bit_words: int,
    sample_size_bits: int,
    num_channels: int,
    is_complex: int,
) -> int:
    """Calculate the number of channel blocks present in a sample block."""
    size_of_channel_block = num_channels * sample_size_bits * 2 ** (is_complex)
    return math.floor(64 * sample_block_length_64_bit_words / size_of_channel_block)


def calc_frame_time_offset(header) -> float:
    """ """
    complete_sample_blocks = calc_complete_sample_blocks_to_end_of_frame(
        header.data_frame_number - 1,
        header.data_array_length,
        header.sample_block_length,
    )
    alignment_period_seconds = header.alignment_period

    offset = (
        complete_sample_blocks
        * alignment_period_seconds
        / header.sample_periods_per_alignment_period
    )

    return offset
