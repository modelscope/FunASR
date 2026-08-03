#include "result-json.h"

#include <stdexcept>

#include "nlohmann/json.hpp"

namespace funasr {
namespace {

nlohmann::json ParseArrayOrEmpty(const std::string& value) {
    if (value.empty()) {
        return nlohmann::json::array();
    }

    auto parsed = nlohmann::json::parse(value, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_array()) {
        return nlohmann::json::array();
    }
    return parsed;
}

}  // namespace

OutputFormat ParseOutputFormat(const std::string& value) {
    if (value == "log") {
        return OutputFormat::kLog;
    }
    if (value == "jsonl") {
        return OutputFormat::kJsonl;
    }
    throw std::invalid_argument("unsupported output format '" + value +
                                "'; expected 'log' or 'jsonl'");
}

std::string FormatResultJson(const std::string& key, const std::string& mode,
                             const std::string& text, const std::string& timestamp,
                             const std::string& stamp_sents) {
    nlohmann::json record = {
        {"key", key},
        {"mode", mode},
        {"text", text},
        {"timestamp", ParseArrayOrEmpty(timestamp)},
        {"stamp_sents", ParseArrayOrEmpty(stamp_sents)},
    };
    return record.dump();
}

}  // namespace funasr
