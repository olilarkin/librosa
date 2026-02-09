#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/core/audio.hpp>
#include <sstream>

namespace cli {

void register_info(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("info", "Show audio file metadata");
    sub->callback([&]() {
        OutputFormatter out(opts.format, opts.precision, opts.no_time, "info", opts.input_file);

        auto sr_native = librosa::get_samplerate(opts.input_file);
        auto duration = librosa::get_duration(opts.input_file);

        auto audio = load_audio(opts);
        auto n_samples = audio.num_samples();
        auto n_channels = audio.num_channels();

        std::ostringstream sr_ss, dur_ss, samp_ss, ch_ss;
        sr_ss << sr_native;
        dur_ss << std::fixed << std::setprecision(opts.precision) << duration;
        samp_ss << n_samples;
        ch_ss << n_channels;

        out.key_value({
            {"filename", opts.input_file},
            {"native_sr", sr_ss.str()},
            {"duration", dur_ss.str()},
            {"samples", samp_ss.str()},
            {"channels", ch_ss.str()}
        });
    });
}

} // namespace cli
