
#ifndef VOCAB_H
#define VOCAB_H

#include <stdint.h>
#include <string>
#include <vector>
#include <map>
#include "nlohmann/json.hpp"
using namespace std;

namespace funasr {
class Vocab {
  private:
    vector<string> vocab;
    std::map<string, int> token_id;
    std::map<string, string> lex_map;
    bool IsEnglish(string ch);
    void LoadVocabFromYaml(const char* filename);
    void LoadVocabFromJson(const char* filename);
    void LoadLex(const char* filename);

  public:
    Vocab(const char *filename);
    Vocab(const char *filename, const char *lex_file);
    ~Vocab();
    int Size() const;
    bool IsChinese(string ch);
    void Vector2String(vector<int> in, std::vector<std::string> &preds);
    string Vector2String(vector<int> in);
    string Vector2StringV2(vector<int> in, std::string language="");
    string Id2String(int id) const;
    string WordFormat(std::string word);
    int GetIdByToken(const std::string &token) const;
    string Word2Lex(const std::string &word) const;
};

} // namespace funasr
#endif
