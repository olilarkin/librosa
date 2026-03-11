#include "librosa/core/audio.hpp"
#include "librosa/core/convert.hpp"
#include "librosa/util/exceptions.hpp"
#include "librosa/util/utils.hpp"
#include <sndfile.h>
#include <cmath>
#include <algorithm>
#include <fftw3.h>

#ifdef LIBROSA_HAS_SOXR
#include <soxr.h>
#endif

namespace librosa {

namespace {
    AudioFileInfo read_audio_info(const std::string& path) {
        SF_INFO sfinfo;
        sfinfo.format = 0;

        SNDFILE* sndfile = sf_open(path.c_str(), SFM_READ, &sfinfo);
        if (!sndfile) {
            throw ParameterError("Failed to open audio file: " + path + " - " + sf_strerror(nullptr));
        }

        AudioFileInfo info{
            static_cast<Eigen::Index>(sfinfo.frames),
            static_cast<Real>(sfinfo.samplerate),
            sfinfo.channels,
            static_cast<Real>(sfinfo.frames) / sfinfo.samplerate
        };

        sf_close(sndfile);
        return info;
    }

    // Helper to compute next power of 2
    int next_power_of_2(int n) {
        int p = 1;
        while (p < n) p *= 2;
        return p;
    }

    // Helper for fast FFT size (simple version)
    int next_fast_fft_size(int n) {
        // For simplicity, use next power of 2
        // A more sophisticated version would find sizes that factor into 2, 3, 5
        return next_power_of_2(n);
    }
}

// ============================================================================
// Audio I/O
// ============================================================================

AudioData load(const std::string& path,
               std::optional<Real> sr,
               bool mono,
               Real offset,
               std::optional<Real> duration) {
    if (offset < 0) {
        throw ParameterError("offset must be non-negative");
    }

    if (duration && *duration < 0) {
        throw ParameterError("duration must be non-negative");
    }

    SF_INFO sfinfo;
    sfinfo.format = 0;

    SNDFILE* sndfile = sf_open(path.c_str(), SFM_READ, &sfinfo);
    if (!sndfile) {
        throw ParameterError("Failed to open audio file: " + path + " - " + sf_strerror(nullptr));
    }

    // Calculate start frame and frame count
    sf_count_t start_frame = static_cast<sf_count_t>(offset * sfinfo.samplerate);
    if (start_frame > sfinfo.frames) {
        sf_close(sndfile);
        throw ParameterError("offset exceeds audio duration for: " + path);
    }

    sf_count_t frame_count = sfinfo.frames - start_frame;

    if (duration) {
        sf_count_t requested_frames = static_cast<sf_count_t>(*duration * sfinfo.samplerate);
        frame_count = std::min(frame_count, requested_frames);
    }

    // Seek to start position
    if (start_frame > 0) {
        sf_seek(sndfile, start_frame, SEEK_SET);
    }

    AudioData result;
    result.sample_rate = static_cast<Real>(sfinfo.samplerate);
    result.channels = mono && sfinfo.channels > 1 ? 1 : sfinfo.channels;

    if (frame_count == 0) {
        result.samples.resize(result.channels, 0);
        sf_close(sndfile);
        if (sr && *sr != result.sample_rate) {
            result.sample_rate = *sr;
        }
        return result;
    }

    // Read audio data
    std::vector<double> buffer(frame_count * sfinfo.channels);
    sf_count_t frames_read = sf_readf_double(sndfile, buffer.data(), frame_count);

    sf_close(sndfile);

    if (frames_read <= 0) {
        throw ParameterError("Failed to read audio data from: " + path);
    }

    // Convert to our format (channels x samples)
    ArrayXXr samples(sfinfo.channels, frames_read);
    for (sf_count_t i = 0; i < frames_read; ++i) {
        for (int c = 0; c < sfinfo.channels; ++c) {
            samples(c, i) = buffer[i * sfinfo.channels + c];
        }
    }

    result.channels = sfinfo.channels;

    // Convert to mono if requested
    if (mono && sfinfo.channels > 1) {
        ArrayXr mono_samples = samples.colwise().mean();
        result.samples.resize(1, mono_samples.size());
        result.samples.row(0) = mono_samples.transpose();
        result.channels = 1;
    } else {
        result.samples = samples;
    }

    // Resample if requested
    if (sr && *sr != result.sample_rate) {
        if (result.channels == 1) {
            ArrayXr resampled = resample(result.samples.row(0), result.sample_rate, *sr);
            result.samples.resize(1, resampled.size());
            result.samples.row(0) = resampled.transpose();
        } else {
            ArrayXXr resampled(result.channels, 0);
            for (int c = 0; c < result.channels; ++c) {
                ArrayXr channel_resampled = resample(result.samples.row(c), result.sample_rate, *sr);
                if (c == 0) {
                    resampled.resize(result.channels, channel_resampled.size());
                }
                resampled.row(c) = channel_resampled.transpose();
            }
            result.samples = resampled;
        }
        result.sample_rate = *sr;
    }

    return result;
}

Real get_duration(const std::string& path) {
    return get_audio_info(path).duration;
}

AudioFileInfo get_audio_info(const std::string& path) {
    return read_audio_info(path);
}

Real get_duration(const ArrayXr& y, Real sr) {
    return static_cast<Real>(y.size()) / sr;
}

Real get_duration(const ArrayXXr& S, Real sr, int hop_length, int n_fft, bool center) {
    Eigen::Index n_frames = S.cols();
    Eigen::Index n_samples;

    if (center) {
        n_samples = (n_frames - 1) * hop_length;
    } else {
        n_samples = (n_frames - 1) * hop_length + n_fft;
    }

    return static_cast<Real>(n_samples) / sr;
}

Real get_samplerate(const std::string& path) {
    return get_audio_info(path).sample_rate;
}

// ============================================================================
// Audio Processing
// ============================================================================

ArrayXr to_mono(const ArrayXXr& y) {
    util::valid_audio(y);

    if (y.rows() == 1) {
        return y.row(0);
    }

    return y.colwise().mean();
}

ArrayXr to_mono(const ArrayXr& y) {
    util::valid_audio(y);
    return y;
}

ArrayXr resample(const ArrayXr& y, Real orig_sr, Real target_sr,
                 const std::string& res_type,
                 bool fix, bool scale) {
    util::valid_audio(y);

    if (orig_sr == target_sr) {
        return y;
    }

    Real ratio = target_sr / orig_sr;
    Eigen::Index n_samples = static_cast<Eigen::Index>(std::ceil(y.size() * ratio));

    ArrayXr y_hat;

#ifdef LIBROSA_HAS_SOXR
    if (res_type.find("soxr") != std::string::npos) {
        // Use soxr for resampling
        soxr_quality_spec_t q_spec;
        if (res_type == "soxr_vhq") {
            q_spec = soxr_quality_spec(SOXR_VHQ, 0);
        } else if (res_type == "soxr_hq") {
            q_spec = soxr_quality_spec(SOXR_HQ, 0);
        } else if (res_type == "soxr_mq") {
            q_spec = soxr_quality_spec(SOXR_MQ, 0);
        } else if (res_type == "soxr_lq") {
            q_spec = soxr_quality_spec(SOXR_LQ, 0);
        } else {
            q_spec = soxr_quality_spec(SOXR_QQ, 0);
        }

        soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT64_I, SOXR_FLOAT64_I);
        soxr_runtime_spec_t runtime_spec = soxr_runtime_spec(1);

        soxr_error_t error;
        soxr_t soxr = soxr_create(orig_sr, target_sr, 1, &error, &io_spec, &q_spec, &runtime_spec);

        if (error) {
            throw ParameterError(std::string("Soxr creation failed: ") + soxr_strerror(error));
        }

        size_t idone, odone;
        std::vector<double> input(y.data(), y.data() + y.size());
        std::vector<double> output(n_samples);

        error = soxr_process(soxr, input.data(), y.size(), &idone,
                             output.data(), n_samples, &odone);

        if (error) {
            soxr_delete(soxr);
            throw ParameterError(std::string("Soxr processing failed: ") + soxr_strerror(error));
        }

        // Drain remaining samples from soxr's internal buffer
        size_t odone2 = 0;
        if (odone < static_cast<size_t>(n_samples)) {
            error = soxr_process(soxr, nullptr, 0, &idone,
                                 output.data() + odone, n_samples - odone, &odone2);
        }

        soxr_delete(soxr);

        if (error) {
            throw ParameterError(std::string("Soxr drain failed: ") + soxr_strerror(error));
        }

        size_t total = odone + odone2;
        y_hat.resize(total);
        for (size_t i = 0; i < total; ++i) {
            y_hat(i) = output[i];
        }
    } else
#endif
    if (res_type == "fft" || res_type == "scipy") {
        // FFT-based resampling
        int n_fft = y.size();
        int n_out = n_samples;

        // Allocate FFTW arrays
        fftw_complex* in_fft = fftw_alloc_complex(n_fft);
        fftw_complex* out_fft = fftw_alloc_complex(n_out);

        // Forward FFT plan
        fftw_plan forward_plan = fftw_plan_dft_r2c_1d(n_fft,
            const_cast<double*>(y.data()), in_fft, FFTW_ESTIMATE);

        // Copy input
        std::vector<double> input(y.data(), y.data() + y.size());

        // Execute forward FFT
        fftw_execute_dft_r2c(forward_plan,
            const_cast<double*>(input.data()), in_fft);

        // Zero-pad or truncate in frequency domain
        int n_freq_in = n_fft / 2 + 1;
        int n_freq_out = n_out / 2 + 1;
        int n_copy = std::min(n_freq_in, n_freq_out);

        for (int i = 0; i < n_freq_out; ++i) {
            if (i < n_copy) {
                out_fft[i][0] = in_fft[i][0];
                out_fft[i][1] = in_fft[i][1];
            } else {
                out_fft[i][0] = 0;
                out_fft[i][1] = 0;
            }
        }

        // Inverse FFT plan
        y_hat.resize(n_out);
        fftw_plan inverse_plan = fftw_plan_dft_c2r_1d(n_out,
            out_fft, y_hat.data(), FFTW_ESTIMATE);

        fftw_execute(inverse_plan);

        // Normalize
        y_hat /= static_cast<Real>(n_fft);

        // Cleanup
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(inverse_plan);
        fftw_free(in_fft);
        fftw_free(out_fft);
    } else if (res_type == "linear") {
        // Simple linear interpolation
        y_hat.resize(n_samples);
        for (Eigen::Index i = 0; i < n_samples; ++i) {
            Real idx = static_cast<Real>(i) / ratio;
            Eigen::Index idx0 = static_cast<Eigen::Index>(idx);
            Eigen::Index idx1 = std::min(idx0 + 1, y.size() - 1);
            Real frac = idx - idx0;
            y_hat(i) = (1.0 - frac) * y(idx0) + frac * y(idx1);
        }
    } else {
        throw ParameterError("Unknown resampling type: " + res_type);
    }

    if (fix) {
        y_hat = util::fix_length(y_hat, n_samples);
    }

    if (scale) {
        y_hat /= std::sqrt(ratio);
    }

    return y_hat;
}

ArrayXr autocorrelate(const ArrayXr& y, std::optional<int> max_size) {
    Eigen::Index n = y.size();
    Eigen::Index max_lag = max_size.value_or(n);
    max_lag = std::min(max_lag, n);

    // Pad for FFT
    int n_fft = next_fast_fft_size(2 * n - 1);

    // Allocate FFTW arrays
    std::vector<double> padded(n_fft, 0.0);
    std::copy(y.data(), y.data() + n, padded.begin());

    fftw_complex* freq = fftw_alloc_complex(n_fft / 2 + 1);

    // Forward FFT
    fftw_plan forward_plan = fftw_plan_dft_r2c_1d(n_fft, padded.data(), freq, FFTW_ESTIMATE);
    fftw_execute(forward_plan);

    // Compute power spectrum
    for (int i = 0; i < n_fft / 2 + 1; ++i) {
        Real re = freq[i][0];
        Real im = freq[i][1];
        freq[i][0] = re * re + im * im;
        freq[i][1] = 0;
    }

    // Inverse FFT
    std::vector<double> autocorr(n_fft);
    fftw_plan inverse_plan = fftw_plan_dft_c2r_1d(n_fft, freq, autocorr.data(), FFTW_ESTIMATE);
    fftw_execute(inverse_plan);

    // Normalize and extract result
    ArrayXr result(max_lag);
    for (Eigen::Index i = 0; i < max_lag; ++i) {
        result(i) = autocorr[i] / n_fft;
    }

    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(freq);

    return result;
}

ArrayXXr autocorrelate(const ArrayXXr& y, std::optional<int> max_size, int axis) {
    if (axis < 0) axis = 2 + axis;

    if (axis == 0) {
        // Along rows
        Eigen::Index n = y.rows();
        Eigen::Index max_lag = max_size.value_or(n);
        max_lag = std::min(max_lag, n);

        ArrayXXr result(max_lag, y.cols());
        for (Eigen::Index j = 0; j < y.cols(); ++j) {
            ArrayXr col = y.col(j);
            result.col(j) = autocorrelate(col, static_cast<int>(max_lag));
        }
        return result;
    } else {
        // Along columns
        Eigen::Index n = y.cols();
        Eigen::Index max_lag = max_size.value_or(n);
        max_lag = std::min(max_lag, n);

        ArrayXXr result(y.rows(), max_lag);
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            ArrayXr row = y.row(i).transpose();
            result.row(i) = autocorrelate(row, static_cast<int>(max_lag)).transpose();
        }
        return result;
    }
}

ArrayXr lpc(const ArrayXr& y, int order) {
    if (order < 1) {
        throw ParameterError("LPC order must be > 0");
    }
    util::valid_audio(y);

    Eigen::Index n = y.size();
    ArrayXr ar_coeffs = ArrayXr::Zero(order + 1);
    ar_coeffs(0) = 1.0;

    ArrayXr ar_coeffs_prev = ar_coeffs;

    // Forward and backward prediction errors
    ArrayXr fwd_pred_error = y.segment(1, n - 1);
    ArrayXr bwd_pred_error = y.segment(0, n - 1);

    // Denominator for reflection coefficient
    Real den = (fwd_pred_error.square() + bwd_pred_error.square()).sum();
    Real epsilon = util::tiny<Real>();

    for (int i = 0; i < order; ++i) {
        // Compute reflection coefficient
        Real reflect_coeff = -2.0 * (bwd_pred_error * fwd_pred_error).sum() / (den + epsilon);

        // Update AR coefficients using Levinson-Durbin recursion
        ar_coeffs_prev = ar_coeffs;
        for (int j = 1; j <= i + 1; ++j) {
            ar_coeffs(j) = ar_coeffs_prev(j) + reflect_coeff * ar_coeffs_prev(i - j + 1);
        }

        // Update prediction errors
        ArrayXr fwd_pred_error_tmp = fwd_pred_error;
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error;
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp;

        // Update denominator
        Real q = 1.0 - reflect_coeff * reflect_coeff;
        den = q * den - bwd_pred_error(bwd_pred_error.size() - 1) * bwd_pred_error(bwd_pred_error.size() - 1)
              - fwd_pred_error(0) * fwd_pred_error(0);

        // Shift prediction errors
        fwd_pred_error = fwd_pred_error.segment(1, fwd_pred_error.size() - 1).eval();
        bwd_pred_error = bwd_pred_error.segment(0, bwd_pred_error.size() - 1).eval();
    }

    return ar_coeffs;
}

ArrayXXr lpc(const ArrayXXr& y, int order, int axis) {
    if (axis < 0) axis = 2 + axis;

    if (axis == 0) {
        ArrayXXr result(order + 1, y.cols());
        for (Eigen::Index j = 0; j < y.cols(); ++j) {
            ArrayXr col = y.col(j);
            result.col(j) = lpc(col, order);
        }
        return result;
    } else {
        ArrayXXr result(y.rows(), order + 1);
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            ArrayXr row = y.row(i).transpose();
            result.row(i) = lpc(row, order).transpose();
        }
        return result;
    }
}

Eigen::Array<bool, Eigen::Dynamic, 1>
zero_crossings(const ArrayXr& y,
               Real threshold,
               std::optional<Real> ref_magnitude,
               bool pad,
               bool zero_pos) {
    Real thresh = threshold;
    if (ref_magnitude) {
        thresh *= *ref_magnitude;
    }

    Eigen::Array<bool, Eigen::Dynamic, 1> result(y.size());
    result(0) = pad;

    for (Eigen::Index i = 1; i < y.size(); ++i) {
        Real x0 = y(i);
        Real x1 = y(i - 1);

        // Apply threshold
        if (-thresh <= x0 && x0 <= thresh) x0 = 0;
        if (-thresh <= x1 && x1 <= thresh) x1 = 0;

        if (zero_pos) {
            result(i) = std::signbit(x0) != std::signbit(x1);
        } else {
            int sign0 = (x0 > 0) ? 1 : ((x0 < 0) ? -1 : 0);
            int sign1 = (x1 > 0) ? 1 : ((x1 < 0) ? -1 : 0);
            result(i) = sign0 != sign1;
        }
    }

    return result;
}

Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
zero_crossings(const ArrayXXr& y,
               Real threshold,
               std::optional<Real> ref_magnitude,
               bool pad,
               bool zero_pos,
               int axis) {
    if (axis < 0) axis = 2 + axis;

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(y.rows(), y.cols());

    if (axis == 0) {
        for (Eigen::Index j = 0; j < y.cols(); ++j) {
            ArrayXr col = y.col(j);
            result.col(j) = zero_crossings(col, threshold, ref_magnitude, pad, zero_pos);
        }
    } else {
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            ArrayXr row = y.row(i).transpose();
            Eigen::Array<bool, Eigen::Dynamic, 1> row_result =
                zero_crossings(row, threshold, ref_magnitude, pad, zero_pos);
            result.row(i) = row_result.transpose();
        }
    }

    return result;
}

// ============================================================================
// Signal Generation
// ============================================================================

ArrayXr clicks(const ArrayXr& times, Real sr,
               Real click_freq, Real click_duration,
               std::optional<int> length) {
    // Generate default click
    int click_length = static_cast<int>(click_duration * sr);
    ArrayXr click(click_length);
    ArrayXr envelope = ArrayXr::LinSpaced(click_length, 1.0, 0.0).square();

    for (int i = 0; i < click_length; ++i) {
        Real t = static_cast<Real>(i) / sr;
        click(i) = envelope(i) * std::sin(2.0 * constants::PI * click_freq * t);
    }

    // Convert times to positions
    std::vector<Eigen::Index> positions;
    for (Eigen::Index i = 0; i < times.size(); ++i) {
        positions.push_back(static_cast<Eigen::Index>(times(i) * sr));
    }

    // Determine output length
    Eigen::Index out_length;
    if (length) {
        out_length = *length;
    } else {
        out_length = positions.back() + click_length;
    }

    // Place clicks
    ArrayXr result = ArrayXr::Zero(out_length);
    for (auto start : positions) {
        if (start >= out_length) continue;

        Eigen::Index end = std::min(start + click_length, out_length);
        Eigen::Index click_end = end - start;
        result.segment(start, click_end) += click.head(click_end);
    }

    return result;
}

ArrayXr clicks_frames(const std::vector<Eigen::Index>& frames, Real sr,
                      int hop_length,
                      Real click_freq, Real click_duration,
                      std::optional<int> length) {
    ArrayXr times(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        times(i) = static_cast<Real>(frames[i] * hop_length) / sr;
    }
    return clicks(times, sr, click_freq, click_duration, length);
}

ArrayXr tone(Real frequency, Real sr,
             std::optional<int> length,
             std::optional<Real> duration,
             std::optional<Real> phi) {
    Eigen::Index n;
    if (length) {
        n = *length;
    } else if (duration) {
        n = static_cast<Eigen::Index>(*duration * sr);
    } else {
        throw ParameterError("Either length or duration must be provided");
    }

    Real phase = phi.value_or(-constants::PI * 0.5);

    ArrayXr result(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        result(i) = std::cos(2.0 * constants::PI * frequency * i / sr + phase);
    }

    return result;
}

ArrayXr chirp(Real fmin, Real fmax, Real sr,
              std::optional<int> length,
              std::optional<Real> duration,
              bool linear,
              std::optional<Real> phi) {
    Real dur;
    if (length) {
        dur = static_cast<Real>(*length) / sr;
    } else if (duration) {
        dur = *duration;
    } else {
        throw ParameterError("Either length or duration must be provided");
    }

    Eigen::Index n = static_cast<Eigen::Index>(dur * sr);
    Real phase = phi.value_or(-constants::PI * 0.5);

    ArrayXr result(n);

    if (linear) {
        // Linear chirp: f(t) = fmin + (fmax - fmin) * t / dur
        Real k = (fmax - fmin) / dur;
        for (Eigen::Index i = 0; i < n; ++i) {
            Real t = static_cast<Real>(i) / sr;
            Real inst_phase = 2.0 * constants::PI * (fmin * t + 0.5 * k * t * t) + phase;
            result(i) = std::cos(inst_phase);
        }
    } else {
        // Exponential (logarithmic) chirp: f(t) = fmin * (fmax/fmin)^(t/dur)
        Real log_ratio = std::log(fmax / fmin);
        for (Eigen::Index i = 0; i < n; ++i) {
            Real t = static_cast<Real>(i) / sr;
            Real inst_phase = 2.0 * constants::PI * fmin * dur / log_ratio *
                             (std::exp(log_ratio * t / dur) - 1.0) + phase;
            result(i) = std::cos(inst_phase);
        }
    }

    return result;
}

// ============================================================================
// Mu-law Compression/Expansion
// ============================================================================

ArrayXr mu_compress(const ArrayXr& x, Real mu, bool quantize) {
    if (mu <= 0) {
        throw ParameterError("mu must be strictly positive");
    }
    if ((x < -1.0).any() || (x > 1.0).any()) {
        throw ParameterError("Input must be in range [-1, 1]");
    }

    ArrayXr x_comp = x.sign() * (1.0 + mu * x.abs()).log() / std::log(1.0 + mu);

    if (quantize) {
        ArrayXr result(x.size());
        int n_levels = static_cast<int>(1 + mu);
        int half_levels = n_levels / 2;

        for (Eigen::Index i = 0; i < x.size(); ++i) {
            // Map from [-1, 1] to [0, n_levels-1], then center around 0
            int level = static_cast<int>((x_comp(i) + 1.0) * 0.5 * (n_levels - 1) + 0.5);
            result(i) = level - half_levels;
        }
        return result;
    }

    return x_comp;
}

ArrayXr mu_expand(const ArrayXr& x, Real mu, bool quantize) {
    if (mu <= 0) {
        throw ParameterError("mu must be strictly positive");
    }

    ArrayXr x_norm = x;
    if (quantize) {
        x_norm = x * 2.0 / (1.0 + mu);
    }

    if ((x_norm < -1.0).any() || (x_norm > 1.0).any()) {
        throw ParameterError("Input must be in range [-1, 1]");
    }

    return x_norm.sign() / mu * (Eigen::pow(1.0 + mu, x_norm.abs()) - 1.0);
}

// Implement mono() method for AudioData
ArrayXr AudioData::mono() const {
    if (channels == 1) {
        return samples.row(0);
    }
    return samples.colwise().mean();
}

} // namespace librosa
