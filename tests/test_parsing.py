from pycodif.parsing import CODIFHeader

import os


class TestCODIFHeader:
    def test_instantiation(self):
        print(os.getcwd())
        header = CODIFHeader("tests/vela_jimble_output.codif")
        assert header.data_frame_number == 477644
        assert header.epoch_offset == 9767380
        assert header.reference_epoch == 49
        assert header.sample_size == 16
        assert header.atypical == 0
        assert header.invalid == 0
        assert header.complex == 1
        assert header.cal_enabled == 0
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

        assert (
            len(header.data_array)
            == header.data_array_length / header.sample_block_length
        )
        assert len(header.data_array[0]) == header.channels
