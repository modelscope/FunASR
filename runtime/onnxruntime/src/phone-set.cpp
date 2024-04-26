#include "phone-set.h"
#include <yaml-cpp/yaml.h>
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>

using namespace std;

namespace funasr {
PhoneSet::PhoneSet(const char *filename) {
  ifstream in(filename);
  LoadPhoneSetFromJson(filename);
}
PhoneSet::~PhoneSet()
{
}

void PhoneSet::LoadPhoneSetFromYaml(const char* filename) {
  YAML::Node config;
  try{
    config = YAML::LoadFile(filename);
  }catch(exception const &e){
     LOG(INFO) << "Error loading file, yaml file error or not exist.";
     exit(-1);
  }
  YAML::Node myList = config["token_list"];
  int id = 0;
  for (YAML::const_iterator it = myList.begin(); it != myList.end(); ++it, id++) {
    phone_.push_back(it->as<string>());
    phn2Id_.emplace(it->as<string>(), id);
  }
}

void PhoneSet::LoadPhoneSetFromJson(const char* filename) {
    nlohmann::json json_array;
    std::ifstream file(filename);
    if (file.is_open()) {
        file >> json_array;
        file.close();
    } else {
        LOG(INFO) << "Error loading token file, token file error or not exist.";
        exit(-1);
    }

    int id = 0;
    for (const auto& element : json_array) {
        phone_.push_back(element);
        phn2Id_.emplace(element, id);
        id++;
    }
}

int PhoneSet::Size() const {
  return phone_.size();
}

int PhoneSet::String2Id(string phn_str) const {
  if (phn2Id_.count(phn_str)) {
    return phn2Id_.at(phn_str);
  } else {
    //LOG(INFO) << "Phone unit not exist.";
    return -1;
  }
}

string PhoneSet::Id2String(int id) const {
  if (id < 0 || id > Size()) {
    //LOG(INFO) << "Phone id not exist.";
    return "";
  } else {
    return phone_[id];
  }
}

bool PhoneSet::Find(string phn_str) const {
  return phn2Id_.count(phn_str) > 0;
}

int PhoneSet::GetBegSilPhnId() const {
  return String2Id(UNIT_BEG_SIL_SYMBOL);
}

int PhoneSet::GetEndSilPhnId() const {
  return String2Id(UNIT_END_SIL_SYMBOL);
}

int PhoneSet::GetBlkPhnId() const {
  return String2Id(UNIT_BLK_SYMBOL);
}

}
