// base/timer.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/timer.h"
#include "base/kaldi-error.h"
#include <algorithm>
#include <iomanip>
#include <map>
#include <unordered_map>

namespace kaldi {

class ProfileStats {
 public:
  void AccStats(const char *function_name, double elapsed) {
    std::unordered_map<const char*, ProfileStatsEntry>::iterator
        iter = map_.find(function_name);
    if (iter == map_.end()) {
      map_[function_name] = ProfileStatsEntry(function_name);
      map_[function_name].total_time = elapsed;
    } else {
      iter->second.total_time += elapsed;
    }
  }
  ~ProfileStats() {
    // This map makes sure we agglomerate the time if there were any duplicate
    // addresses of strings.
    std::unordered_map<std::string, double> total_time;
    for (auto iter = map_.begin(); iter != map_.end(); iter++)
      total_time[iter->second.name] += iter->second.total_time;

    ReverseSecondComparator comp;
    std::vector<std::pair<std::string, double> > pairs(total_time.begin(),
                                                       total_time.end());
    std::sort(pairs.begin(), pairs.end(), comp);
    for (size_t i = 0; i < pairs.size(); i++) {
      KALDI_LOG << "Time taken in " << pairs[i].first << " is "
                << std::fixed << std::setprecision(2) << pairs[i].second << "s.";
    }
  }
 private:

  struct ProfileStatsEntry {
    std::string name;
    double total_time;
    ProfileStatsEntry() { }
    ProfileStatsEntry(const char *name): name(name) { }
  };

  struct ReverseSecondComparator {
    bool operator () (const std::pair<std::string, double> &a,
                      const std::pair<std::string, double> &b) {
      return a.second > b.second;
    }
  };

  // Note: this map is keyed on the address of the string, there is no proper
  // hash function.  The assumption is that the strings are compile-time
  // constants.
  std::unordered_map<const char*, ProfileStatsEntry> map_;
};

ProfileStats g_profile_stats;

Profiler::~Profiler() {
  g_profile_stats.AccStats(name_, tim_.Elapsed());
}

}  // namespace kaldi
