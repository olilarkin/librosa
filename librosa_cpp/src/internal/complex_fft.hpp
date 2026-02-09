#pragma once

#include <librosa/types.hpp>
#include <fftw3.h>
#include <algorithm>
#include <cstring>

namespace librosa {
namespace internal {

/// Compute FFT of each row of a complex matrix, keeping only non-negative frequencies.
/// Input rows are zero-padded to n_fft if shorter.
/// @param input Complex array (n_rows, input_cols)
/// @param n_fft FFT size (must be >= input_cols)
/// @return Complex array (n_rows, n_fft/2 + 1)
inline ArrayXXc complex_fft_rows(const ArrayXXc& input, int n_fft) {
    Eigen::Index n_rows = input.rows();
    int n_out = n_fft / 2 + 1;

    ArrayXXc result(n_rows, n_out);

    // Allocate FFTW buffers
    fftw_complex* fft_in = fftw_alloc_complex(n_fft);
    fftw_complex* fft_out = fftw_alloc_complex(n_fft);
    fftw_plan plan = fftw_plan_dft_1d(n_fft, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (Eigen::Index r = 0; r < n_rows; ++r) {
        // Zero the buffer
        std::memset(fft_in, 0, sizeof(fftw_complex) * n_fft);

        // Copy row data (RowMajor: row data is contiguous)
        Eigen::Index cols = std::min(input.cols(), static_cast<Eigen::Index>(n_fft));
        for (Eigen::Index c = 0; c < cols; ++c) {
            fft_in[c][0] = input(r, c).real();
            fft_in[c][1] = input(r, c).imag();
        }

        // Execute FFT
        fftw_execute(plan);

        // Copy non-negative frequencies to output
        for (int c = 0; c < n_out; ++c) {
            result(r, c) = Complex(fft_out[c][0], fft_out[c][1]);
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);

    return result;
}

} // namespace internal
} // namespace librosa
