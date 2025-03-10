import numpy as np
import pytest

from pycodif.date_functions import (
    calc_complete_sample_blocks_to_end_of_frame,
    calc_time_of_all_samples_in_frame,
)
from pycodif.parsing import CODIFHeader


class TestCalcCompleteSampleBlocks:
    @pytest.fixture
    def header(self):
        with open("tests/test_files/test_codif.codif", "rb") as f:
            header = CODIFHeader(f)

        return header

    def test_first_frame_has_zero_complete(self, header):
        header.data_frame_number = 0

        complete_sample_blocks = calc_complete_sample_blocks_to_end_of_frame(
            header.data_frame_number,
            header.data_array_length,
            header.sample_block_length,
        )

        assert complete_sample_blocks == 64

    def test_time_of_all_samples_in_frame(self, header):
        header.data_frame_number = 0

        time_of_samples = calc_time_of_all_samples_in_frame(header)

        assert isinstance(time_of_samples, np.ndarray)
        assert len(time_of_samples) == 64
        assert np.isclose(time_of_samples[0], 0)
