#include <string>
#include <vector>
#include <sstream>
#include <cctype>
#include <cstdio>
#include <algorithm>

// 辅助函数：修剪字符串两端的空白
std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t");
    if (start == std::string::npos) return "";
    
    auto end = s.find_last_not_of(" \t");
    return s.substr(start, end - start + 1);
}

// 辅助函数：检查字符串是否以某个前缀开头
bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && 
           s.compare(0, prefix.size(), prefix) == 0;
}

// 辅助函数：分割字符串
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream token_stream(s);
    while (std::getline(token_stream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

// URL 解码（简易实现）
std::string url_decode(const std::string& str) {
    std::string result;
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '%' && i + 2 < str.size()) {
            int hex_val;
            if (sscanf(str.substr(i + 1, 2).c_str(), "%x", &hex_val) == 1) {
                result += static_cast<char>(hex_val);
                i += 2;
            } else {
                result += str[i];
            }
        } else if (str[i] == '+') {
            result += ' ';
        } else {
            result += str[i];
        }
    }
    return result;
}

// 解析 RFC 5987 编码（如 filename*=UTF-8''%C2%A3.txt）
std::string decode_rfc5987(const std::string& value) {
    size_t pos = value.find("''");
    if (pos != std::string::npos) {
        std::string encoded = value.substr(pos + 2);
        return url_decode(encoded);
    }
    return value;
}

// 主解析函数
std::string parse_attachment_filename_impl(const std::string& content_disp) {
    std::vector<std::string> parts = split(content_disp, ';');
    std::string filename;

    for (auto& part : parts) {
        std::string trimmed = trim(part);
        
        // 优先处理 RFC 5987 编码的 filename*
        if (starts_with(trimmed, "filename*=")) {
            std::string value = trimmed.substr(10);
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
            return decode_rfc5987(value);
        }
        
        // 其次处理普通 filename
        else if (starts_with(trimmed, "filename=")) {
            std::string value = trimmed.substr(9);
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
            filename = value;
        }
    }

    return filename;
}