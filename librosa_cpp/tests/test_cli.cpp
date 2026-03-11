#include <gtest/gtest.h>
#include "../cli/common.hpp"
#include "../cli/formatter.hpp"
#include <librosa/util/exceptions.hpp>
#include <sndfile.h>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace cli;

namespace {

class TempAudioFile {
public:
    TempAudioFile(int sample_rate, int channels, sf_count_t frames) {
        char path_template[] = "/tmp/librosa-cli-XXXXXX.wav";
        int fd = mkstemps(path_template, 4);
        if (fd == -1) {
            throw std::runtime_error("Failed to create temporary audio path");
        }

        ::close(fd);
        path_ = path_template;

        SF_INFO sfinfo;
        sfinfo.frames = frames;
        sfinfo.samplerate = sample_rate;
        sfinfo.channels = channels;
        sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        sfinfo.sections = 0;
        sfinfo.seekable = 0;

        SNDFILE* sndfile = sf_open(path_.c_str(), SFM_WRITE, &sfinfo);
        if (!sndfile) {
            throw std::runtime_error("Failed to open temporary audio file for writing");
        }

        std::vector<double> buffer(frames * channels, 0.5);
        sf_count_t written = sf_writef_double(sndfile, buffer.data(), frames);
        sf_close(sndfile);

        if (written != frames) {
            throw std::runtime_error("Failed to write temporary audio file");
        }
    }

    ~TempAudioFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

} // namespace

TEST(OutputFormatterTest, JsonEscapesStringFields) {
    OutputFormatter formatter("json", 6, false, "info\"cmd", "C:\\tmp\\\"a\"\n.wav");

    testing::internal::CaptureStdout();
    formatter.key_value({
        {"filename", "C:\\tmp\\\"a\"\n.wav"},
        {"note", "123abc"},
        {"channels", "2"}
    });
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("\"file\": \"C:\\\\tmp\\\\\\\"a\\\"\\n.wav\""), std::string::npos);
    EXPECT_NE(output.find("\"command\": \"info\\\"cmd\""), std::string::npos);
    EXPECT_NE(output.find("\"filename\": \"C:\\\\tmp\\\\\\\"a\\\"\\n.wav\""), std::string::npos);
    EXPECT_NE(output.find("\"note\": \"123abc\""), std::string::npos);
    EXPECT_NE(output.find("\"channels\": 2"), std::string::npos);
}

TEST(OutputFormatterTest, JsonUsesNullForNonFiniteScalars) {
    OutputFormatter formatter("json", 6, false, "tempo", "file.wav");

    testing::internal::CaptureStdout();
    formatter.scalar("tempo", std::numeric_limits<double>::quiet_NaN());
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("\"tempo\": null"), std::string::npos);
}

TEST(CommonOptionsTest, RejectsNegativeDurationExceptSentinel) {
    TempAudioFile file(22050, 1, 128);

    CommonOptions opts;
    opts.input_file = file.path();
    opts.duration = -0.5;

    EXPECT_THROW(load_audio(opts), librosa::ParameterError);
}
