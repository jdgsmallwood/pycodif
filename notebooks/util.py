from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.fft import fft, fftshift


def get_fourier_power_spectrum(data: np.ndarray, segment_size: int) -> np.ndarray:
    window = np.hanning(segment_size)

    n_segments = (data.shape[1] - segment_size) // (segment_size)

    output = np.zeros((n_segments, segment_size))

    for j in range(n_segments):
        start = j * (segment_size)
        end = start + segment_size
        if end > data.shape[1]:
            print("past end of array")
            break
        sliced_data = data[0, start:end]

        ff_transform = fft(sliced_data * window)
        ff_shift = fftshift(ff_transform)
        power = np.abs(ff_shift)

        output[j, :] = power

    return output


def get_demeaned_scrunched_data(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) - data.mean(axis=1).mean(axis=0, keepdims=True)


def calc_phase_delay_radians(
    theta_radians: float,
    phi_radians: float,
    x_diff_m: float,
    y_diff_m: float,
    reference_wavelength: float,
) -> float:
    wave_vector = np.array(
        [
            np.cos(theta_radians) * np.cos(phi_radians),
            np.sin(theta_radians) * np.cos(phi_radians),
            np.sin(phi_radians),
        ]
    )

    # Does this need to be in u/v coordinates? theta would be measured from the x-axis and phi from the plane to the vector?
    # I don't think it matters so long as it's consistent (and we know the orientation of the PAF).
    baseline_m = np.array([x_diff_m, y_diff_m, 0])

    phase_delay_radians = (
        np.dot(wave_vector, baseline_m) * 2 * np.pi / reference_wavelength
    )

    return phase_delay_radians


def calc_phase_shift_for_antennae(
    theta_radians: float,
    phi_radians: float,
    antennae_map: pd.DataFrame,
    reference_wavelength: float,
) -> np.ndarray:
    output = np.zeros(len(antennae_map))
    for antenna in antennae_map.itertuples():
        output[antenna.Index] = np.exp(
            calc_phase_delay_radians(
                theta_radians,
                phi_radians,
                antenna.x_loc,
                antenna.y_loc,
                reference_wavelength,
            )
        )

    return output


def get_fold_segment(
    period_ms: float, seconds_between_samples: float, segment_size: int
) -> int:
    samples_in_period = period_ms / 1000 / seconds_between_samples

    fold_segment = int(round(samples_in_period / segment_size, 0))
    return fold_segment


def get_folded_data(
    data: np.ndarray,
    n_bins: int,
    period_ms: float,
    seconds_between_samples: float,
    segment_size: int,
) -> np.ndarray:
    """
    segment_size is the number of samples in window.
    """
    seconds_in_window = seconds_between_samples * segment_size
    output_folded = np.zeros((n_bins, segment_size))
    period_s = period_ms / 1000
    cumulative_time = np.arange(data.shape[0]) * seconds_in_window
    bin_to_add_to = (
        (((cumulative_time % period_s) / period_s) * n_bins).round(0) % n_bins
    ).astype(int)
    num_folds = defaultdict(int)

    for i in range(data.shape[0]):
        output_folded[bin_to_add_to[i]] += data[i]
        num_folds[bin_to_add_to[i]] += 1

    for j in range(output_folded.shape[0]):
        if num_folds[j] != 0:
            output_folded[j] /= num_folds[j]

    return output_folded


def dedisperse_dataset(
    data: np.ndarray,
    dm: float,
    seconds_between_samples: float,
    frequencies_axis_mhz: np.ndarray,
    segment_size: int,
) -> np.ndarray:
    SECONDS_BETWEEN_SEGMENTS = seconds_between_samples * (segment_size)

    delay_samples = np.round(
        4.15
        * 10**3
        * dm
        * ((frequencies_axis_mhz**-2) - (frequencies_axis_mhz[-1] ** -2))
        / SECONDS_BETWEEN_SEGMENTS,
        0,
    ).astype(int)

    transposed_data = np.transpose(data, (1, 0)).copy()
    transposed_data_shape = transposed_data.shape

    final_data = np.zeros(
        shape=(
            transposed_data_shape[0],
            transposed_data_shape[1] - max(delay_samples),
        )
    )
    for i in range(delay_samples.shape[0]):
        if delay_samples[i] < max(delay_samples):
            final_data[i, :] = transposed_data[
                i, delay_samples[i] : (delay_samples[i] - max(delay_samples))
            ]
        else:
            final_data[i, :] = transposed_data[i, delay_samples[i] :]
    dedispersed_data = np.transpose(final_data, (1, 0))
    return dedispersed_data
