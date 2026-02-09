#include "formatter.hpp"
#include <iomanip>
#include <sstream>
#include <cmath>

namespace cli {

OutputFormatter::OutputFormatter(const std::string& format, int precision, bool no_time,
                                 const std::string& command, const std::string& file)
    : format_(format), precision_(precision), no_time_(no_time),
      command_(command), file_(file) {}

std::string OutputFormatter::fmt(double v) const {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision_) << v;
    return oss.str();
}

void OutputFormatter::scalar(const std::string& label, double value) {
    auto& os = std::cout;
    if (format_ == "json") {
        os << "{\n";
        os << "  \"file\": \"" << file_ << "\",\n";
        os << "  \"command\": \"" << command_ << "\",\n";
        os << "  \"" << label << "\": " << fmt(value) << "\n";
        os << "}\n";
    } else if (format_ == "csv") {
        os << label << "\n";
        os << fmt(value) << "\n";
    } else {
        os << fmt(value) << "\n";
    }
}

void OutputFormatter::key_value(const std::vector<std::pair<std::string, std::string>>& pairs) {
    auto& os = std::cout;
    if (format_ == "json") {
        os << "{\n";
        os << "  \"file\": \"" << file_ << "\",\n";
        os << "  \"command\": \"" << command_ << "\"";
        for (auto& [k, v] : pairs) {
            os << ",\n  \"" << k << "\": ";
            // Try to detect if it's a number
            bool is_num = false;
            try {
                std::stod(v);
                is_num = true;
            } catch (...) {}
            if (is_num) os << v;
            else os << "\"" << v << "\"";
        }
        os << "\n}\n";
    } else if (format_ == "csv") {
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (i > 0) os << ",";
            os << pairs[i].first;
        }
        os << "\n";
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (i > 0) os << ",";
            os << pairs[i].second;
        }
        os << "\n";
    } else {
        for (auto& [k, v] : pairs) {
            os << k << "\t" << v << "\n";
        }
    }
}

void OutputFormatter::vector(const librosa::ArrayXr& data, const std::string& label) {
    auto& os = std::cout;
    if (format_ == "json") {
        os << "{\n";
        os << "  \"file\": \"" << file_ << "\",\n";
        os << "  \"command\": \"" << command_ << "\",\n";
        os << "  \"count\": " << data.size() << ",\n";
        os << "  \"" << label << "\": [";
        for (Eigen::Index i = 0; i < data.size(); ++i) {
            if (i > 0) os << ", ";
            os << fmt(data(i));
        }
        os << "]\n}\n";
    } else if (format_ == "csv") {
        os << label << "\n";
        for (Eigen::Index i = 0; i < data.size(); ++i) {
            os << fmt(data(i)) << "\n";
        }
    } else {
        for (Eigen::Index i = 0; i < data.size(); ++i) {
            os << fmt(data(i)) << "\n";
        }
    }
}

void OutputFormatter::per_frame(const librosa::ArrayXr& times, const librosa::ArrayXXr& data,
                                const std::vector<std::string>& labels) {
    // data shape: (n_features, n_frames) — librosa convention
    Eigen::Index n_frames = data.cols();
    Eigen::Index n_feat = data.rows();
    auto& os = std::cout;

    if (format_ == "json") {
        os << "{\n";
        os << "  \"file\": \"" << file_ << "\",\n";
        os << "  \"command\": \"" << command_ << "\",\n";
        os << "  \"shape\": [" << n_frames << ", " << n_feat << "],\n";
        if (!no_time_) {
            os << "  \"times\": [";
            for (Eigen::Index t = 0; t < n_frames; ++t) {
                if (t > 0) os << ", ";
                os << fmt(times(t));
            }
            os << "],\n";
        }
        os << "  \"data\": [\n";
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            os << "    [";
            for (Eigen::Index f = 0; f < n_feat; ++f) {
                if (f > 0) os << ", ";
                os << fmt(data(f, t));
            }
            os << "]";
            if (t < n_frames - 1) os << ",";
            os << "\n";
        }
        os << "  ]\n}\n";
    } else if (format_ == "csv") {
        // Header
        if (!no_time_) os << "time,";
        for (Eigen::Index f = 0; f < n_feat; ++f) {
            if (f > 0) os << ",";
            os << (f < (Eigen::Index)labels.size() ? labels[f] : "c" + std::to_string(f));
        }
        os << "\n";
        // Data
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            if (!no_time_) os << fmt(times(t)) << ",";
            for (Eigen::Index f = 0; f < n_feat; ++f) {
                if (f > 0) os << ",";
                os << fmt(data(f, t));
            }
            os << "\n";
        }
    } else {
        // Text: tab-separated, one line per frame
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            if (!no_time_) os << fmt(times(t));
            for (Eigen::Index f = 0; f < n_feat; ++f) {
                if (!no_time_ || f > 0) os << "\t";
                os << fmt(data(f, t));
            }
            os << "\n";
        }
    }
}

void OutputFormatter::matrix(const librosa::ArrayXr& times, const librosa::ArrayXXr& data,
                             const std::vector<std::string>& col_labels) {
    // Alias to per_frame — same layout
    per_frame(times, data, col_labels);
}

void OutputFormatter::pitch_output(const librosa::ArrayXr& times, const librosa::ArrayXr& f0,
                                   const librosa::ArrayXr& voiced_prob) {
    Eigen::Index n = f0.size();
    auto& os = std::cout;

    if (format_ == "json") {
        os << "{\n";
        os << "  \"file\": \"" << file_ << "\",\n";
        os << "  \"command\": \"" << command_ << "\",\n";
        os << "  \"count\": " << n << ",\n";
        if (!no_time_) {
            os << "  \"times\": [";
            for (Eigen::Index i = 0; i < n; ++i) {
                if (i > 0) os << ", ";
                os << fmt(times(i));
            }
            os << "],\n";
        }
        os << "  \"f0\": [";
        for (Eigen::Index i = 0; i < n; ++i) {
            if (i > 0) os << ", ";
            if (std::isnan(f0(i))) os << "null";
            else os << fmt(f0(i));
        }
        os << "],\n";
        os << "  \"voiced_probability\": [";
        for (Eigen::Index i = 0; i < n; ++i) {
            if (i > 0) os << ", ";
            os << fmt(voiced_prob(i));
        }
        os << "]\n}\n";
    } else if (format_ == "csv") {
        if (!no_time_) os << "time,";
        os << "f0,voiced_probability\n";
        for (Eigen::Index i = 0; i < n; ++i) {
            if (!no_time_) os << fmt(times(i)) << ",";
            os << fmt(f0(i)) << "," << fmt(voiced_prob(i)) << "\n";
        }
    } else {
        for (Eigen::Index i = 0; i < n; ++i) {
            if (!no_time_) os << fmt(times(i)) << "\t";
            os << fmt(f0(i)) << "\t" << fmt(voiced_prob(i)) << "\n";
        }
    }
}

} // namespace cli
