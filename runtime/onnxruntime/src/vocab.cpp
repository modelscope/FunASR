#include "vocab.h"
#include <yaml-cpp/yaml.h>
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>

using namespace std;

namespace funasr {
Vocab::Vocab(const char *filename)
{
    ifstream in(filename);
    LoadVocabFromJson(filename);
}
Vocab::Vocab(const char *filename, const char *lex_file)
{
    ifstream in(filename);
    LoadVocabFromYaml(filename);
    LoadLex(lex_file);
}
Vocab::~Vocab()
{
}

void Vocab::LoadVocabFromYaml(const char* filename){
    YAML::Node config;
    try{
        config = YAML::LoadFile(filename);
    }catch(exception const &e){
        LOG(INFO) << "Error loading file, yaml file error or not exist.";
        exit(-1);
    }
    YAML::Node myList = config["token_list"];
    int i = 0;
    for (YAML::const_iterator it = myList.begin(); it != myList.end(); ++it) {
        vocab.push_back(it->as<string>());
        token_id[it->as<string>()] = i;
        i ++;
    }
}

void Vocab::LoadVocabFromJson(const char* filename){
    nlohmann::json json_array;
    std::ifstream file(filename);
    if (file.is_open()) {
        file >> json_array;
        file.close();
    } else {
        LOG(INFO) << "Error loading token file, token file error or not exist.";
        exit(-1);
    }

    int i = 0;
    for (const auto& element : json_array) {
        vocab.push_back(element);
        token_id[element] = i;
        i++;
    }
}

void Vocab::LoadLex(const char* filename){
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::string key, value;
        std::istringstream iss(line);
        std::getline(iss, key, '\t');
        std::getline(iss, value);

        if (!key.empty() && !value.empty()) {
            lex_map[key] = value;
        }
    }

    file.close();
}

string Vocab::Word2Lex(const std::string &word) const {
    auto it = lex_map.find(word);
    if (it != lex_map.end()) {
        return it->second;
    }
    return "";
}

int Vocab::GetIdByToken(const std::string &token) const {
    auto it = token_id.find(token);
    if (it != token_id.end()) {
        return it->second;
    }
    return -1;
}

void Vocab::Vector2String(vector<int> in, std::vector<std::string> &preds)
{
    for (auto it = in.begin(); it != in.end(); it++) {
        string word = vocab[*it];
        preds.emplace_back(word);
    }
}

string Vocab::Vector2String(vector<int> in)
{
    int i;
    stringstream ss;
    for (auto it = in.begin(); it != in.end(); it++) {
        ss << vocab[*it];
    }
    return ss.str();
}

int Str2Int(string str)
{
    const char *ch_array = str.c_str();
    if (((ch_array[0] & 0xf0) != 0xe0) || ((ch_array[1] & 0xc0) != 0x80) ||
        ((ch_array[2] & 0xc0) != 0x80))
        return 0;
    int val = ((ch_array[0] & 0x0f) << 12) | ((ch_array[1] & 0x3f) << 6) |
              (ch_array[2] & 0x3f);
    return val;
}

string Vocab::Id2String(int id) const
{
  if (id < 0 || id >= vocab.size()) {
    LOG(INFO) << "Error vocabulary id, this id do not exit.";
    return "";
  } else {
    return vocab[id];
  }
}

bool Vocab::IsChinese(string ch)
{
    if (ch.size() != 3) {
        return false;
    }
    int unicode = Str2Int(ch);
    if (unicode >= 19968 && unicode <= 40959) {
        return true;
    }
    return false;
}

string Vocab::WordFormat(std::string word)
{
    if(word == "i"){
        return "I";
    }else if(word == "i'm"){
        return "I'm";
    }else if(word == "i've"){
        return "I've";
    }else if(word == "i'll"){
        return "I'll";
    }else{
        return word;
    }
}

string Vocab::Vector2StringV2(vector<int> in, std::string language)
{
    int i;
    list<string> words;
    int is_pre_english = false;
    int pre_english_len = 0;
    int is_combining = false;
    std::string combine = "";
    std::string unicodeChar = "▁";

    for (i=0; i<in.size(); i++){
        string word = vocab[in[i]];
        // step1 space character skips
        if (word == "<s>" || word == "</s>" || word == "<unk>")
            continue;
        if (language == "en-bpe"){
            size_t found = word.find(unicodeChar);
            if(found != std::string::npos){
                if (combine != ""){
                    combine = WordFormat(combine);
                    if (words.size() != 0){
                        combine = " " + combine;
                    }
                    words.push_back(combine);
                }
                combine = word.substr(3);
            }else{
                combine += word;
            }
            continue;
        }
        // step2 combie phoneme to full word
        {
            int sub_word = !(word.find("@@") == string::npos);
            // process word start and middle part
            if (sub_word) {
                // if badcase: lo@@ chinese
                if (i == in.size()-1 || i<in.size()-1 && IsChinese(vocab[in[i+1]])){
                    word = word.erase(word.length() - 2) + " ";
                    if (is_combining) {
                        combine += word;
                        is_combining = false;
                        word = combine;
                        combine = "";
                    }
                }else{
                    combine += word.erase(word.length() - 2);
                    is_combining = true;
                    continue;
                }
            }
            // process word end part
            else if (is_combining) {
                combine += word;
                is_combining = false;
                word = combine;
                combine = "";
            }
        }

        // step3 process english word deal with space , turn abbreviation to upper case
        {
            // input word is chinese, not need process 
            if (IsChinese(word)) {
                words.push_back(word);
                is_pre_english = false;
            }
            // input word is english word
            else {
                // pre word is chinese
                if (!is_pre_english) {
                    // word[0] = word[0] - 32;
                    words.push_back(word);
                    pre_english_len = word.size();
                }
                // pre word is english word
                else {
                    // single letter turn to upper case
                    // if (word.size() == 1) {
                    //     word[0] = word[0] - 32;
                    // }

                    if (pre_english_len > 1) {
                        words.push_back(" ");
                        words.push_back(word);
                        pre_english_len = word.size();
                    } 
                    else {
                        if (word.size() > 1) {
                            words.push_back(" ");
                        }
                        words.push_back(word);
                        pre_english_len = word.size();
                    }
                }
                is_pre_english = true;
            }
        }
    }

    if (language == "en-bpe" && combine != ""){
        combine = WordFormat(combine);
        if (words.size() != 0){
            combine = " " + combine;
        }
        words.push_back(combine);
    }

    stringstream ss;
    for (auto it = words.begin(); it != words.end(); it++) {
        ss << *it;
    }

    return ss.str();
}

int Vocab::Size() const
{
    return vocab.size();
}

} // namespace funasr
