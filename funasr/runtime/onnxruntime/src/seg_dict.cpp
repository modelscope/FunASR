/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#include "precomp.h"
//#include "util.h"
//#include "seg_dict.h"
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>

using namespace std;

namespace funasr {
SegDict::SegDict(const char *filename)
{
    ifstream in(filename);
    if (!in) {
      LOG(ERROR) << filename << " open failed !!";
      return;
    }
    string textline;
    while (getline(in, textline)) {
      std::vector<string> line_item = split(textline, '\t');
      //std::cout << textline << std::endl;
      if (line_item.size() > 1) {
        std::string word = line_item[0];
        std::string segs = line_item[1];
        std::vector<string> segs_vec = split(segs, ' ');
        seg_dict[word] = segs_vec;
      }
    }
    LOG(INFO) << "load seg dict successfully";
}
std::vector<std::string> SegDict::GetTokensByWord(const std::string &word) {
  if (seg_dict.count(word))
    return seg_dict[word];
  else {
    std::vector<string> vec;
    return vec;
  }
}

SegDict::~SegDict()
{
}


} // namespace funasr
