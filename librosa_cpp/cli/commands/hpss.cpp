#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/effects.hpp>
#include <sndfile.h>
#include <stdexcept>
#include <iostream>
#include <memory>

namespace cli {

static void write_wav(const std::string& path, const librosa::ArrayXr& y, double sr) {
    SF_INFO sfinfo;
    sfinfo.samplerate = static_cast<int>(sr);
    sfinfo.channels = 1;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* sndfile = sf_open(path.c_str(), SFM_WRITE, &sfinfo);
    if (!sndfile) {
        throw std::runtime_error("Failed to open output file: " + path + " - " + sf_strerror(nullptr));
    }

    // Convert to double buffer for sf_writef_double
    std::vector<double> buffer(y.size());
    Eigen::Map<Eigen::VectorXd>(buffer.data(), y.size()) = y.matrix().cast<double>();

    sf_writef_double(sndfile, buffer.data(), y.size());
    sf_close(sndfile);
}

void register_hpss(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("hpss", "Harmonic/percussive source separation (writes WAV)");

    auto output_path = std::make_shared<std::string>();
    auto component = std::make_shared<std::string>("harmonic");
    sub->add_option("--output,-o", *output_path, "Output WAV file path")->required();
    sub->add_option("--component", *component, "Component: harmonic or percussive")->default_val("harmonic");

    sub->callback([&opts, output_path, component]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        librosa::ArrayXr result;
        if (*component == "percussive") {
            result = librosa::effects::percussive(y);
        } else {
            result = librosa::effects::harmonic(y);
        }

        write_wav(*output_path, result, sr);
        std::cerr << "Wrote " << *component << " component to " << *output_path << "\n";
    });
}

} // namespace cli
