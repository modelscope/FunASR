#pragma once

#include <stdexcept>
#include <string>

namespace funasr {

enum class OutputFormat {
    kLog,
    kJsonl,
};

OutputFormat ParseOutputFormat(const std::string& value);

std::string FormatResultJson(const std::string& key, const std::string& mode,
                             const std::string& text, const std::string& timestamp,
                             const std::string& stamp_sents);

}  // namespace funasr
