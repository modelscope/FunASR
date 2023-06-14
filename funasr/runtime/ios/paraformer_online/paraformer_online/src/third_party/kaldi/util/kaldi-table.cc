// util/kaldi-table.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "util/kaldi-table.h"
#include "util/text-utils.h"

namespace kaldi {


bool ReadScriptFile(const std::string &rxfilename,
                    bool warn,
                    std::vector<std::pair<std::string, std::string> >
                    *script_out) {
  bool is_binary;
  Input input;

  if (!input.Open(rxfilename, &is_binary)) {
    if (warn) KALDI_WARN << "Error opening script file: " <<
                 PrintableRxfilename(rxfilename);
    return false;
  }
  if (is_binary) {
    if (warn) KALDI_WARN << "Error: script file appears to be binary: " <<
                 PrintableRxfilename(rxfilename);
    return false;
  }

  bool ans = ReadScriptFile(input.Stream(), warn, script_out);
  if (warn && !ans)
    KALDI_WARN << "[script file was: " << PrintableRxfilename(rxfilename) <<
                  "]";
  return ans;
}

bool ReadScriptFile(std::istream &is,
                    bool warn,
                    std::vector<std::pair<std::string, std::string> >
                    *script_out) {
  KALDI_ASSERT(script_out != NULL);
  std::string line;
  int line_number = 0;
  while (getline(is, line)) {
    line_number++;
    const char *c = line.c_str();
    if (*c == '\0') {
      if (warn)
        KALDI_WARN << "Empty " << line_number << "'th line in script file";
      return false;  // Empty line so invalid scp file format..
    }

    std::string key, rest;
    SplitStringOnFirstSpace(line, &key, &rest);

    if (key.empty() || rest.empty()) {
      if (warn)
        KALDI_WARN << "Invalid " << line_number << "'th line in script file"
                          <<":\"" << line << '"';
      return false;
    }
    script_out->resize(script_out->size()+1);
    script_out->back().first = key;
    script_out->back().second = rest;
  }
  return true;
}

bool WriteScriptFile(std::ostream &os,
                     const std::vector<std::pair<std::string, std::string> >
                     &script) {
  if (!os.good()) {
    KALDI_WARN << "WriteScriptFile: attempting to write to invalid stream.";
    return false;
  }
  std::vector<std::pair<std::string, std::string> >::const_iterator iter;
  for (iter = script.begin(); iter != script.end(); ++iter) {
    if (!IsToken(iter->first)) {
      KALDI_WARN << "WriteScriptFile: using invalid token \"" << iter->first <<
                    '"';
      return false;
    }
    if (iter->second.find('\n') != std::string::npos ||
       (iter->second.length() != 0 &&
        (isspace(iter->second[0]) ||
         isspace(iter->second[iter->second.length()-1])))) {
      // second part contains newline or leading or trailing space.
      KALDI_WARN << "WriteScriptFile: attempting to write invalid line \"" <<
                    iter->second << '"';
      return false;
    }
    os << iter->first << ' ' << iter->second << '\n';
  }
  if (!os.good()) {
    KALDI_WARN << "WriteScriptFile: stream in error state.";
    return false;
  }
  return true;
}

bool WriteScriptFile(const std::string &wxfilename,
                     const std::vector<std::pair<std::string, std::string> >
                     &script) {
  Output output;
  if (!output.Open(wxfilename, false, false)) {  // false, false means not
    // binary, no binary-mode header.
    KALDI_ERR << "Error opening output stream for script file: "
              << PrintableWxfilename(wxfilename);
    return false;
  }
  if (!WriteScriptFile(output.Stream(), script)) {
    KALDI_ERR << "Error writing script file to stream "
              << PrintableWxfilename(wxfilename);
    return false;
  }
  return true;
}



WspecifierType ClassifyWspecifier(const std::string &wspecifier,
                                  std::string *archive_wxfilename,
                                  std::string *script_wxfilename,
                                  WspecifierOptions *opts) {
  //  Examples:
  //  ark,t:wxfilename -> kArchiveWspecifier
  //  ark,b:wxfilename -> kArchiveWspecifier
  //  scp,t:rxfilename -> kScriptWspecifier
  //  scp,t:rxfilename -> kScriptWspecifier
  //  ark,scp,t:filename, wxfilename -> kBothWspecifier
  //  ark,scp:filename, wxfilename ->  kBothWspecifier
  //  Note we can include the flush option (f) or no-flush (nf)
  // anywhere: e.g.
  //  ark,scp,f:filename, wxfilename ->  kBothWspecifier
  // or:
  //  scp,t,nf:rxfilename -> kScriptWspecifier

  if (archive_wxfilename) archive_wxfilename->clear();
  if (script_wxfilename) script_wxfilename->clear();

  size_t pos = wspecifier.find(':');
  if (pos == std::string::npos) return kNoWspecifier;
  if (isspace(*(wspecifier.rbegin()))) return kNoWspecifier;  // Trailing space
  // disallowed.

  std::string before_colon(wspecifier, 0, pos), after_colon(wspecifier, pos+1);

  std::vector<std::string> split_first_part;  // Split part before ':' on ', '.
  SplitStringToVector(before_colon, ", ", false, &split_first_part);  // false==
  // don't omit empty strings between commas.

  WspecifierType ws = kNoWspecifier;

  if (opts != NULL)
    *opts = WspecifierOptions();  // Make sure all the defaults are as in the
                                  // default constructor of the options class.

  for (size_t i = 0; i < split_first_part.size(); i++) {
    const std::string &str = split_first_part[i];  // e.g. "b", "t", "f", "ark",
    // "scp".
    const char *c = str.c_str();
    if (!strcmp(c, "b")) {
      if (opts) opts->binary = true;
    } else if (!strcmp(c, "f")) {
      if (opts) opts->flush = true;
    } else if (!strcmp(c, "nf")) {
      if (opts) opts->flush = false;
    } else if (!strcmp(c, "t")) {
      if (opts) opts->binary = false;
    } else if (!strcmp(c, "p")) {
      if (opts) opts->permissive = true;
    } else if (!strcmp(c, "ark")) {
      if (ws == kNoWspecifier) ws = kArchiveWspecifier;
      else
        return kNoWspecifier;  // We do not allow "scp, ark", only "ark,
      // scp".
    } else if (!strcmp(c, "scp")) {
      if (ws == kNoWspecifier) ws = kScriptWspecifier;
      else if (ws == kArchiveWspecifier) ws = kBothWspecifier;
      else
        return kNoWspecifier;  // repeated "scp" option: invalid.
    } else {
      return kNoWspecifier;  // Could not interpret this option.
    }
  }

  switch (ws) {
    case kArchiveWspecifier:
      if (archive_wxfilename)
        *archive_wxfilename = after_colon;
      break;
    case kScriptWspecifier:
      if (script_wxfilename)
        *script_wxfilename = after_colon;
      break;
    case kBothWspecifier:
      pos = after_colon.find(',');  // first comma.
      if (pos == std::string::npos) return kNoWspecifier;
      if (archive_wxfilename)
        *archive_wxfilename = std::string(after_colon, 0, pos);
      if (script_wxfilename)
        *script_wxfilename = std::string(after_colon, pos+1);
      break;
    case kNoWspecifier: default: break;
  }
  return ws;
}



RspecifierType ClassifyRspecifier(const std::string &rspecifier,
                                  std::string *wxfilename,
                                  RspecifierOptions *opts) {
  // Examples
  // ark:rxfilename  ->  kArchiveRspecifier
  // scp:rxfilename  -> kScriptRspecifier
  //
  // We also allow the meaningless prefixes b, and t,
  // plus the options o (once), no (not-once),
  // s (sorted) and ns (not-sorted), p (permissive)
  // and np (not-permissive).
  // so the following would be valid:
  //
  // f, o, b, np, ark:rxfilename  ->  kArchiveRspecifier
  //
  // Examples:
  //
  // b, ark:rxfilename  ->  kArchiveRspecifier
  // t, ark:rxfilename  ->  kArchiveRspecifier
  // b, scp:rxfilename  -> kScriptRspecifier
  // t, no, s, scp:rxfilename  -> kScriptRspecifier
  // t, ns, scp:rxfilename  -> kScriptRspecifier

  // Improperly formed Rspecifiers will be classified as kNoRspecifier.

  if (wxfilename) wxfilename->clear();

  if (opts != NULL)
    *opts = RspecifierOptions();  // Make sure all the defaults are as in the
                                  // default constructor of the options class.

  size_t pos = rspecifier.find(':');
  if (pos == std::string::npos) return kNoRspecifier;

  if (isspace(*(rspecifier.rbegin()))) return kNoRspecifier;  // Trailing space
  // disallowed.

  std::string before_colon(rspecifier, 0, pos),
      after_colon(rspecifier, pos+1);

  std::vector<std::string> split_first_part;  // Split part before ':' on ', '.
  SplitStringToVector(before_colon, ", ", false, &split_first_part);  // false==
  // don't omit empty strings between commas.

  RspecifierType rs = kNoRspecifier;

  for (size_t i = 0; i < split_first_part.size(); i++) {
    const std::string &str = split_first_part[i];  // e.g. "b", "t", "f", "ark",
    // "scp".
    const char *c = str.c_str();
    if (!strcmp(c, "b"));  // Ignore this option.  It's so we can use the same
    // specifiers for rspecifiers and wspecifiers.
    else if (!strcmp(c, "t"));  // Ignore this option too.
    else if (!strcmp(c, "o")) {
      if (opts) opts->once = true;
    } else if (!strcmp(c, "no")) {
      if (opts) opts->once = false;
    } else if (!strcmp(c, "p")) {
      if (opts) opts->permissive = true;
    } else if (!strcmp(c, "np")) {
      if (opts) opts->permissive = false;
    } else if (!strcmp(c, "s")) {
      if (opts) opts->sorted = true;
    } else if (!strcmp(c, "ns")) {
      if (opts) opts->sorted = false;
    } else if (!strcmp(c, "cs")) {
      if (opts) opts->called_sorted = true;
    } else if (!strcmp(c, "ncs")) {
      if (opts) opts->called_sorted = false;
    } else if (!strcmp(c, "bg")) {
      if (opts) opts->background = true;
    } else if (!strcmp(c, "ark")) {
      if (rs == kNoRspecifier) rs = kArchiveRspecifier;
      else
        return kNoRspecifier;  // Repeated or combined ark and scp options
      // invalid.
    } else if (!strcmp(c, "scp")) {
      if (rs == kNoRspecifier) rs = kScriptRspecifier;
      else
        return kNoRspecifier;  // Repeated or combined ark and scp options
      // invalid.
    } else {
      return kNoRspecifier;  // Could not interpret this option.
    }
  }
  if ((rs == kArchiveRspecifier || rs == kScriptRspecifier)
     && wxfilename != NULL)
    *wxfilename = after_colon;
  return rs;
}






}  // end namespace kaldi
