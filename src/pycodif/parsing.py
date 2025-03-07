import struct
import math
import numpy as np


class CODIFHeader:
    def __init__(self, filename: str):
        self.filename = filename
        with open(self.filename, "rb") as f:
            self.parse_header(f)

            self.effective_sample_size = self.sample_size * 2**self.complex  # bits
            self.channel_block_size_bits = self.effective_sample_size  # bits
            self.channel_block_size_bytes = self.channel_block_size_bits / 8
            self.channel_blocks_per_sample_block = math.floor(
                self.sample_block_length * 64 / self.channel_block_size_bits
            )

            self.read_data(f)

    def parse_header(self, f):
        byte_data = f.read(4)
        self.data_frame_number = int(struct.unpack("<I", byte_data)[0])

        byte_data = f.read(4)
        self.epoch_offset = int(struct.unpack("<I", byte_data)[0])

        byte_data = f.read(8)
        byte_0, byte_1, packed_bits_1, packed_bits_2, byte4, byte5, byte6, byte7 = (
            struct.unpack("<8B", byte_data)
        )
        self.reference_epoch = int(byte_0)
        self.sample_size = int(byte_1)

        print(packed_bits_1)
        self.sample_representation = int(packed_bits_1 & 0xF)  # this will be bits 4-7
        self.cal_enabled = int((packed_bits_1 >> 3) & 0x1)
        self.complex = int((packed_bits_1 >> 2) & 0x1)
        self.invalid = int((packed_bits_1 >> 1) & 0x1)
        self.atypical = int((packed_bits_1 >> 0) & 0x1)

        self.protocol = int(packed_bits_2 >> 5 & 0xF)
        self.version = int(packed_bits_2 & 0x1F)

        # bytes 4-5 are reserved for future use so do nothing with them
        self.reserved = int((byte5 << 8) | byte4)

        # bytes 6-7 are the alignment period
        self.alignment_period = int((byte7 << 8) | byte6)

        # next word
        # 0-1 Thread ID
        # 2-3 Group ID
        # 4-5 Secondary ID
        # 6-7 Station ID

        byte_code = f.read(8)
        (
            self.thread_id,
            self.group_id,
            self.secondary_id,
            self.station_id_1,
            self.station_id_2,
        ) = struct.unpack("<3HBB", byte_code)
        self.station_id = f"{chr(self.station_id_1)}{chr(self.station_id_2)}"

        # next word
        # 0-1 Channels
        # 2-3 Sample Block Length in units of 8 bytes
        # 4-7 Data array length in units of 8 bytes
        byte_code = f.read(8)
        self.channels, self.sample_block_length, self.data_array_length = struct.unpack(
            "<HHI", byte_code
        )

        # Word 4
        # 0-7 sample periods per alignment period
        byte_code = f.read(8)
        self.sample_periods_per_alignment_period = struct.unpack("<Q", byte_code)[0]

        # word 5
        # 0-3 synchronisation sequence
        # 4-5 metadata id
        # 6-7 metadata bytes 0-1

        byte_code = f.read(6)
        self.synchronisation_sequence, self.metadata_id = struct.unpack(
            "<IH", byte_code
        )

        byte_code = f.read(18)
        self.metadata_bytes = struct.unpack("<18B", byte_code)

    def read_data(self, f):
        data_array_bytes = f.read(8 * self.data_array_length)
        number_of_samples = int(self.data_array_length / self.sample_block_length)
        # samples = data_array_bytes / self.sample_block_length

        sample_groups = []
        for sample in range(number_of_samples):
            sample_start = sample * self.sample_block_length * 8
            sample_end = sample_start + self.sample_block_length * 8
            sample_data = data_array_bytes[sample_start:sample_end]

            channels = []
            for channel in range(self.channels):
                channel_start = channel * int(self.channel_block_size_bytes)
                channel_end = channel_start + int(self.channel_block_size_bytes)
                channel_data = sample_data[channel_start:channel_end]

                real_part, imag_part = struct.unpack("<2H", channel_data)
                channels.append((real_part, imag_part))
            sample_groups.append(channels)
        # sample_blocks = struct.unpack(struct_string, data_array_bytes)
        self.data_array = np.array(sample_groups)
