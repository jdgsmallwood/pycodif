from datetime import datetime

import numpy as np

from pycodif.parsing import CODIF, CODIFFrame, CODIFHeader


class TestCODIFHeader:
    def test_instantiation(self):
        with open("tests/test_files/test_codif.codif", "rb") as f:
            header = CODIFHeader(f)
            assert header.data_frame_number == 477644
            assert header.epoch_offset == 9767380
            assert header.reference_epoch == 49
            assert header.sample_size == 16
            assert header.atypical == 0
            assert header.invalid == 0
            assert header.complex == 1
            assert header.cal_enabled == 0

            # Revisit this one - it should be 1.
            assert header.sample_representation == 4

            assert header.version == 3
            assert header.protocol == 7
            assert header.alignment_period == 27

            assert header.thread_id == 215
            assert header.group_id == 17
            assert header.secondary_id == 926
            assert header.station_id == "KP"

            assert header.channels == 8
            assert header.sample_block_length == 4
            assert header.data_array_length == 256
            assert header.sample_periods_per_alignment_period == 51200000

            assert hex(header.synchronisation_sequence) == "0xfeedcafe"
            assert header.metadata_id == 0

            assert all(meta_byte == 0 for meta_byte in header.metadata_bytes)
            assert len(header.metadata_bytes) == 18

            assert header.channel_block_size_bits == 16 * 2
            assert header.channel_block_size_bytes == 16 * 2 / 8
            assert header.channel_blocks_per_sample_block == header.channels

    def test_timestamp(self):
        with open("tests/test_files/test_codif.codif", "rb") as f:
            header = CODIFHeader(f)
            assert header.epoch_date == datetime(2024, 7, 1, 0, 0, 0)
            # Yet to be known if this is the correct datetime for the observation - need to cross-check with AJ
            assert header.start_alignment_period_timestamp == datetime(
                2024, 10, 22, 1, 9, 40
            )

            assert np.isclose(header.frame_time_offset, 16.120, atol=0.001)
            assert header.start_frame_timestamp == datetime(
                2024, 10, 22, 1, 9, 56, 120485
            )


class TestCODIFFrame:
    def test_data_parsing(self):
        with open("tests/test_files/test_codif.codif", "rb") as f:
            codif = CODIFFrame(f)
        assert hasattr(codif, "header")
        assert hasattr(codif, "data_array")
        assert hasattr(codif, "sample_timestamps")

    def test_data_values(self):
        with open("tests/test_files/test_codif.codif", "rb") as f:
            codif = CODIFFrame(f)

        assert isinstance(codif.data_array, np.ndarray)
        assert codif.data_array.dtype == np.dtype("complex64")
        assert codif.data_array[0, 0] == -23 + 45j
        assert codif.data_array[0, -1] == 113 - 89j
        assert codif.data_array[-1, 0] == 45j
        assert codif.data_array[-1, -1] == -43 + 58j

    # assert isinstance(codif.data_array, np.array)
    # assert (
    #     len(header.data_array)
    #     == header.data_array_length / header.sample_block_length
    # )
    # assert len(header.data_array[0]) == header.channels


class TestCODIF:
    def test_parsing(self):
        codif = CODIF("tests/test_files/test_codif.codif")
        assert hasattr(codif, "frames")

    def test_timestamps(self):
        codif = CODIF("tests/test_files/test_codif.codif")
        assert hasattr(codif, "timestamps")
        assert isinstance(codif.timestamps, np.ndarray)
        assert len(codif.timestamps.shape) == 1
