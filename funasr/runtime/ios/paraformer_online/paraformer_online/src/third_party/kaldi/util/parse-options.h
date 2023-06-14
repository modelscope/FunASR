// util/parse-options.h

// Copyright 2009-2011  Karel Vesely;  Microsoft Corporation;
//                      Saarland University (Author: Arnab Ghoshal);
// Copyright 2012-2013  Frantisek Skala;  Arnab Ghoshal

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_UTIL_PARSE_OPTIONS_H_
#define KALDI_UTIL_PARSE_OPTIONS_H_

#include <map>
#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "itf/options-itf.h"

namespace kaldi {

/// The class ParseOptions is for parsing command-line options; see
/// \ref parse_options for more documentation.
class ParseOptions : public OptionsItf {
 public:
  explicit ParseOptions(const char *usage) :
    print_args_(true), help_(false), usage_(usage), argc_(0), argv_(NULL),
    prefix_(""), other_parser_(NULL) {
#if !defined(_MSC_VER) && !defined(__CYGWIN__) // This is just a convenient place to set the stderr to line
    setlinebuf(stderr);  // buffering mode, since it's called at program start.
#endif  // This helps ensure different programs' output is not mixed up.
    RegisterStandard("config", &config_, "Configuration file to read (this "
                     "option may be repeated)");
    RegisterStandard("print-args", &print_args_,
                     "Print the command line arguments (to stderr)");
    RegisterStandard("help", &help_, "Print out usage message");
    RegisterStandard("verbose", &g_kaldi_verbose_level,
                     "Verbose level (higher->more logging)");
  }

  /**
    This is a constructor for the special case where some options are
    registered with a prefix to avoid conflicts.  The object thus created will
    only be used temporarily to register an options class with the original
    options parser (which is passed as the *other pointer) using the given
    prefix.  It should not be used for any other purpose, and the prefix must
    not be the empty string.  It seems to be the least bad way of implementing
    options with prefixes at this point.
    Example of usage is:
     ParseOptions po;  // original ParseOptions object
     ParseOptions po_mfcc("mfcc", &po); // object with prefix.
     MfccOptions mfcc_opts;
     mfcc_opts.Register(&po_mfcc);
    The options will now get registered as, e.g., --mfcc.frame-shift=10.0
    instead of just --frame-shift=10.0
   */
  ParseOptions(const std::string &prefix, OptionsItf *other);

  ~ParseOptions() {}

  // Methods from the interface
  void Register(const std::string &name,
                bool *ptr, const std::string &doc);
  void Register(const std::string &name,
                int32 *ptr, const std::string &doc);
  void Register(const std::string &name,
                uint32 *ptr, const std::string &doc);
  void Register(const std::string &name,
                float *ptr, const std::string &doc);
  void Register(const std::string &name,
                double *ptr, const std::string &doc);
  void Register(const std::string &name,
                std::string *ptr, const std::string &doc);

  /// If called after registering an option and before calling
  /// Read(), disables that option from being used.  Will crash
  /// at runtime if that option had not been registered.
  void DisableOption(const std::string &name);

  /// This one is used for registering standard parameters of all the programs
  template<typename T>
  void RegisterStandard(const std::string &name,
                        T *ptr, const std::string &doc);

  /**
    Parses the command line options and fills the ParseOptions-registered
    variables. This must be called after all the variables were registered!!!

    Initially the variables have implicit values,
    then the config file values are set-up,
    finally the command line vaues given.
    Returns the first position in argv that was not used.
    [typically not useful: use NumParams() and GetParam(). ]
   */
  int Read(int argc, const char *const *argv);

  /// Prints the usage documentation [provided in the constructor].
  void PrintUsage(bool print_command_line = false);
  /// Prints the actual configuration of all the registered variables
  void PrintConfig(std::ostream &os);

  /// Reads the options values from a config file.  Must be called after
  /// registering all options.  This is usually used internally after the
  /// standard --config option is used, but it may also be called from a
  /// program.
  void ReadConfigFile(const std::string &filename);

  /// Number of positional parameters (c.f. argc-1).
  int NumArgs() const;

  /// Returns one of the positional parameters; 1-based indexing for argc/argv
  /// compatibility. Will crash if param is not >=1 and <=NumArgs().
  std::string GetArg(int param) const;

  std::string GetOptArg(int param) const {
    return (param <= NumArgs() ? GetArg(param) : "");
  }

  /// The following function will return a possibly quoted and escaped
  /// version of "str", according to the current shell.  Currently
  /// this is just hardwired to bash.  It's useful for debug output.
  static std::string Escape(const std::string &str);

 private:
  /// Template to register various variable types,
  /// used for program-specific parameters
  template<typename T>
  void RegisterTmpl(const std::string &name, T *ptr, const std::string &doc);

  // Following functions do just the datatype-specific part of the job
  /// Register boolean variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        bool *b, const std::string &doc, bool is_standard);
  /// Register int32 variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        int32 *i, const std::string &doc, bool is_standard);
  /// Register unsinged  int32 variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        uint32 *u,
                        const std::string &doc, bool is_standard);
  /// Register float variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        float *f, const std::string &doc, bool is_standard);
  /// Register double variable [useful as we change BaseFloat type].
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        double *f, const std::string &doc, bool is_standard);
  /// Register string variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        std::string *s, const std::string &doc,
                        bool is_standard);

  /// Does the actual job for both kinds of parameters
  /// Does the common part of the job for all datatypes,
  /// then calls RegisterSpecific
  template<typename T>
  void RegisterCommon(const std::string &name,
                      T *ptr, const std::string &doc, bool is_standard);

  /// Set option with name "key" to "value"; will crash if can't do it.
  /// "has_equal_sign" is used to allow --x for a boolean option x,
  /// and --y=, for a string option y.
  bool SetOption(const std::string &key, const std::string &value,
                 bool has_equal_sign);

  bool ToBool(std::string str);
  int32 ToInt(const std::string &str);
  uint32 ToUint(const std::string &str);
  float ToFloat(const std::string &str);
  double ToDouble(const std::string &str);

  // maps for option variables
  std::map<std::string, bool*> bool_map_;
  std::map<std::string, int32*> int_map_;
  std::map<std::string, uint32*> uint_map_;
  std::map<std::string, float*> float_map_;
  std::map<std::string, double*> double_map_;
  std::map<std::string, std::string*> string_map_;

  /**
     Structure for options' documentation
   */
  struct DocInfo {
    DocInfo() {}
    DocInfo(const std::string &name, const std::string &usemsg)
      : name_(name), use_msg_(usemsg), is_standard_(false) {}
    DocInfo(const std::string &name, const std::string &usemsg,
            bool is_standard)
      : name_(name), use_msg_(usemsg),  is_standard_(is_standard) {}

    std::string name_;
    std::string use_msg_;
    bool is_standard_;
  };
  typedef std::map<std::string, DocInfo> DocMapType;
  DocMapType doc_map_;  ///< map for the documentation

  bool print_args_;     ///< variable for the implicit --print-args parameter
  bool help_;           ///< variable for the implicit --help parameter
  std::string config_;  ///< variable for the implicit --config parameter
  std::vector<std::string> positional_args_;
  const char *usage_;
  int argc_;
  const char *const *argv_;

  /// These members are not normally used. They are only used when the object
  /// is constructed with a prefix
  std::string prefix_;
  OptionsItf *other_parser_;
 protected:
    /// SplitLongArg parses an argument of the form --a=b, --a=, or --a,
  /// and sets "has_equal_sign" to true if an equals-sign was parsed..
  /// this is needed in order to correctly allow --x for a boolean option
  /// x, and --y= for a string option y, and to disallow --x= and --y.
  void SplitLongArg(std::string in, std::string *key, std::string *value,
                    bool *has_equal_sign);

  void NormalizeArgName(std::string *str);
};

/// This template is provided for convenience in reading config classes from
/// files; this is not the standard way to read configuration options, but may
/// occasionally be needed.  This function assumes the config has a function
/// "void Register(OptionsItf *opts)" which it can call to register the
/// ParseOptions object.
template<class C> void ReadConfigFromFile(const std::string config_filename,
                                          C *c) {
  std::ostringstream usage_str;
  usage_str << "Parsing config from "
            << "from '" << config_filename << "'";
  ParseOptions po(usage_str.str().c_str());
  c->Register(&po);
  po.ReadConfigFile(config_filename);
}

/// This variant of the template ReadConfigFromFile is for if you need to read
/// two config classes from the same file.
template<class C1, class C2> void ReadConfigsFromFile(const std::string
                                                      config_filename,
                                                      C1 *c1, C2 *c2) {
  std::ostringstream usage_str;
  usage_str << "Parsing config from "
            << "from '" << config_filename << "'";
  ParseOptions po(usage_str.str().c_str());
  c1->Register(&po);
  c2->Register(&po);
  po.ReadConfigFile(config_filename);
}



}  // namespace kaldi

#endif  // KALDI_UTIL_PARSE_OPTIONS_H_
