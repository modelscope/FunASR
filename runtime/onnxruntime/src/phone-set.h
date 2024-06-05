#ifndef PHONESET_H
#define PHONESET_H

#include <stdint.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "nlohmann/json.hpp"
#define UNIT_BEG_SIL_SYMBOL "<s>"
#define UNIT_END_SIL_SYMBOL "</s>"
#define UNIT_BLK_SYMBOL "<blank>"

using namespace std;

namespace funasr {
class PhoneSet {
  public:
    PhoneSet(const char *filename);
    ~PhoneSet();
    int Size() const;
    int String2Id(string str) const;
    string Id2String(int id) const;
    bool Find(string str) const;
    int GetBegSilPhnId() const;
    int GetEndSilPhnId() const;
    int GetBlkPhnId() const;

  private:
    vector<string> phone_;
    unordered_map<string, int> phn2Id_;
    void LoadPhoneSetFromYaml(const char* filename);
    void LoadPhoneSetFromJson(const char* filename);
};

} // namespace funasr
#endif
