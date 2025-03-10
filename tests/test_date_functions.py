from pycodif.date_functions import calc_complete_sample_blocks_to_end_of_frame
from pycodif.parsing import CODIFHeader


class TestCalcCompleteSampleBlocks:
    def test_first_frame_has_zero_complete(self):
        with open("tests/test_files/test_codif.codif", "rb") as f:
            header = CODIFHeader(f)

        header.data_frame_number = 0

        complete_sample_blocks = calc_complete_sample_blocks_to_end_of_frame(
            header.data_frame_number,
            header.data_array_length,
            header.sample_block_length,
        )

        assert complete_sample_blocks == 64
