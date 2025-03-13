import math
import struct
from collections import defaultdict
from datetime import timedelta

import numpy as np
from loguru import logger
from tqdm import tqdm

from pycodif.date_functions import (
    calc_epoch_base,
    calc_frame_time_offset,
    calc_start_alignment_period_timestamp,
    calc_time_of_all_samples_in_frame,
)


class CODIFHeader:
    def __init__(self, f):
        self.parse_header(f)

        self.effective_sample_size = self.sample_size * 2**self.complex  # bits
        self.channel_block_size_bits = self.effective_sample_size  # bits
        self.channel_block_size_bytes = self.channel_block_size_bits / 8
        self.channel_blocks_per_sample_block = math.floor(
            self.sample_block_length * 64 / self.channel_block_size_bits
        )

        self.epoch_date = calc_epoch_base(self.reference_epoch)
        self.start_alignment_period_timestamp = calc_start_alignment_period_timestamp(
            self.epoch_date, self.epoch_offset
        )

        self.frame_time_offset = calc_frame_time_offset(self)
        self.start_frame_timestamp = self.start_alignment_period_timestamp + timedelta(
            seconds=self.frame_time_offset
        )

    def parse_header(self, f):
        byte_data = f.read(8)
        self.data_frame_number, self.epoch_offset = struct.unpack("<II", byte_data)

        byte_data = f.read(8)
        byte_0, byte_1, packed_bits_1, packed_bits_2, byte4, byte5, byte6, byte7 = (
            struct.unpack("<8B", byte_data)
        )
        self.reference_epoch = int(byte_0)
        self.sample_size = int(byte_1)

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
        # Word 4
        # 0-7 sample periods per alignment period
        byte_code = f.read(16)
        (
            self.channels,
            self.sample_block_length,
            self.data_array_length,
            self.sample_periods_per_alignment_period,
        ) = struct.unpack("<HHIQ", byte_code)

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


class CODIFFrame:
    def __init__(self, f):
        self.header = CODIFHeader(f)
        self.sample_timestamps = calc_time_of_all_samples_in_frame(self.header)
        self.read_data(f)

    def read_data(self, f):
        data_array_bytes = f.read(8 * self.header.data_array_length)
        self.number_of_samples = int(
            self.header.data_array_length / self.header.sample_block_length
        )

        # this will be little endian by default
        data_array = np.frombuffer(data_array_bytes, dtype=np.short).astype(float)
        data_array = data_array.reshape(self.number_of_samples, self.header.channels, 2)
        data_array = data_array.transpose(1, 0, 2)

        self.data_array = (data_array[..., 0] + 1j * data_array[..., 1]).astype(
            np.complex64
        )


class CODIF:
    def __init__(self, filename: str):
        self.frames = {}

        logger.info("Starting file decode...")
        with open(filename, "rb") as f:
            while f.read(1):
                f.seek(-1, 1)  # if not end of file - then reset to before the f.read()
                frame = CODIFFrame(f)

                self.frames[
                    (
                        frame.header.data_frame_number,
                        frame.header.thread_id,
                        frame.header.group_id,
                        frame.header.secondary_id,
                        frame.header.station_id,
                    )
                ] = frame
        logger.info("Finished reading file, converting....")
        self.datasets = defaultdict(lambda: defaultdict(list))

        sorted_keys = sorted(self.frames.keys(), key=lambda x: x[0])

        all_stations = set(a[4] for a in self.frames.keys())
        all_frames = set([a[0] for a in self.frames.keys()])
        all_threads = set([a[1] for a in self.frames.keys()])
        all_groups = set([a[2] for a in self.frames.keys()])
        # This will get used at some point but comment out for now
        # all_secondaries = set([a[3] for a in self.frames.keys()])

        array_shape_frames = len(all_frames)
        array_shape_threads = len(all_threads)
        array_shape_group = len(all_groups)

        # array_shape_secondary = len(set([a[3] for a in self.frames.keys()]))
        array_shape_station = len(all_stations)
        array_shape_channels = self.frames[sorted_keys[0]].header.channels
        array_shape_samples = (
            self.frames[sorted_keys[0]].number_of_samples * array_shape_frames
        )

        array_shape = (
            array_shape_station,
            array_shape_group,
            array_shape_threads,
            array_shape_channels,
            array_shape_samples,
        )

        logger.info("getting timestamps")
        min_threads = min([a[1] for a in self.frames.keys()])
        min_group = min([a[2] for a in self.frames.keys() if a[1] == min_threads])
        min_secondary = min(
            [
                a[3]
                for a in self.frames.keys()
                if a[1] == min_threads and a[2] == min_group
            ]
        )
        min_station = min(
            [
                a[4]
                for a in self.frames.keys()
                if a[1] == min_threads and a[2] == min_group and a[3] == min_secondary
            ]
        )

        # Make an assumption that all of the timestamps are the same for now. So we
        # only have to get the timestamps from one thread group.
        # Sort these as it is theoretically possible that these are out of order.
        keys_for_timestamp = sorted(
            [
                a
                for a in self.frames.keys()
                if a[1] == min_threads
                and a[2] == min_group
                and a[3] == min_secondary
                and a[4] == min_station
            ],
            key=lambda x: x[0],
        )

        timestamps = []
        for key in tqdm(keys_for_timestamp):
            timestamps.append(self.frames[key].sample_timestamps)
        self.timestamps = np.concatenate(timestamps)

        self.data = np.empty(array_shape, dtype=np.complex64)
        logger.info("Concatenating data frames...")
        for i, station in enumerate(all_stations):
            for j, group in enumerate(all_groups):
                for k, thread in enumerate(all_threads):
                    self.data[i, j, k] = np.concatenate(
                        [
                            self.frames[id].data_array
                            for id in [
                                a
                                for a in sorted_keys
                                if a[1] == thread and a[2] == group and a[4] == station
                            ]
                        ],
                        axis=1,
                    )

        logger.info("Finished writing array!")
