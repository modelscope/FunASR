/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

#ifndef VOCAB_H
#define VOCAB_H

#include <stdint.h>
#include <string>
#include <vector>
using namespace std;

namespace funasr {
class Vocab {
  private:
    vector<string> vocab;
    bool IsChinese(string ch);
    bool IsEnglish(string ch);
    void LoadVocabFromYaml(const char* filename);

  public:
    Vocab(const char *filename);
    ~Vocab();
    int Size();
    string Vector2String(vector<int> in);
    string Vector2StringV2(vector<int> in);
};

} // namespace funasr
#endif
