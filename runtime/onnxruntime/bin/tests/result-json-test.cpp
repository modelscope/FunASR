#include "result-json.h"

#include <cassert>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

int main() {
    using funasr::FormatResultJson;
    using funasr::OutputFormat;
    using funasr::ParseOutputFormat;

    assert(ParseOutputFormat("log") == OutputFormat::kLog);
    assert(ParseOutputFormat("jsonl") == OutputFormat::kJsonl);

    bool rejected_invalid_format = false;
    try {
        ParseOutputFormat("yaml");
    } catch (const std::invalid_argument&) {
        rejected_invalid_format = true;
    }
    assert(rejected_invalid_format);

    const auto record = nlohmann::json::parse(FormatResultJson(
        "utt-\"1", "offline", "你好 \"FunASR\"", "[[0,320],[320,660]]",
        "[{\"text_seg\":\"你好\",\"start\":0,\"end\":660}]"));
    assert(record.at("key") == "utt-\"1");
    assert(record.at("mode") == "offline");
    assert(record.at("text") == "你好 \"FunASR\"");
    assert(record.at("timestamp") == nlohmann::json::parse("[[0,320],[320,660]]"));
    assert(record.at("stamp_sents").at(0).at("start") == 0);

    const auto empty_record = nlohmann::json::parse(
        FormatResultJson("utt-2", "2pass", "done", "", "not-json"));
    assert(empty_record.at("timestamp") == nlohmann::json::array());
    assert(empty_record.at("stamp_sents") == nlohmann::json::array());

    const auto wrong_type_record = nlohmann::json::parse(
        FormatResultJson("utt-3", "online", "partial", "{\"start\":0}", "null"));
    assert(wrong_type_record.at("timestamp") == nlohmann::json::array());
    assert(wrong_type_record.at("stamp_sents") == nlohmann::json::array());

    return 0;
}
