#include "vocab.h"
#include <yaml-cpp/yaml.h>
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>

using namespace std;

Vocab::Vocab(const char *filename)
{
    ifstream in(filename);
    LoadVocabFromYaml(filename);
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
    for (YAML::const_iterator it = myList.begin(); it != myList.end(); ++it) {
        vocab.push_back(it->as<string>());
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

string Vocab::Vector2StringV2(vector<int> in)
{
    int i;
    list<string> words;
    int is_pre_english = false;
    int pre_english_len = 0;
    int is_combining = false;
    string combine = "";

    for (auto it = in.begin(); it != in.end(); it++) {
        string word = vocab[*it];
        // step1 space character skips
        if (word == "<s>" || word == "</s>" || word == "<unk>")
            continue;
        // step2 combie phoneme to full word
        {
            int sub_word = !(word.find("@@") == string::npos);
            // process word start and middle part
            if (sub_word) {
                combine += word.erase(word.length() - 2);
                is_combining = true;
                continue;
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
                    word[0] = word[0] - 32;
                    words.push_back(word);
                    pre_english_len = word.size();

                }
                // pre word is english word
                else {
                    // single letter turn to upper case
                    if (word.size() == 1) {
                        word[0] = word[0] - 32;
                    }

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

    stringstream ss;
    for (auto it = words.begin(); it != words.end(); it++) {
        ss << *it;
    }

    return ss.str();
}

int Vocab::Size()
{
    return vocab.size();
}
