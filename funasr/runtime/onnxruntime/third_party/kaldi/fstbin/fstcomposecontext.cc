// fstbin/fstcomposecontext.cc

// Copyright 2009-2011  Microsoft Corporation

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/context-fst.h"
#include "fstext/grammar-context-fst.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

/*
  A couple of test examples:

  pushd ~/tmpdir
  # (1) with no disambig syms.
  ( echo "0 1 1 1"; echo "1 2 2 2"; echo "2 3 3 3"; echo "3 0" ) | fstcompile | fstcomposecontext ilabels.sym > tmp.fst
  ( echo "<eps> 0"; echo "a 1"; echo "b 2"; echo "c 3" ) > phones.txt
  fstmakecontextsyms phones.txt ilabels.sym > context.txt
  fstprint --isymbols=context.txt --osymbols=phones.txt tmp.fst
  # and the result is:

WARNING (fstcomposecontext[5.4]:main():fstcomposecontext.cc:130) Disambiguation symbols list is empty; this likely indicates an error in data preparation.
0	1	<eps>	a
1	2	<eps>/a/b	b
2	3	a/b/c	c
3	4	b/c/<eps>	<eps>
4


  # (2) with disambig syms:
  ( echo 4; echo 5) > disambig.list
  ( echo "<eps> 0"; echo "a 1"; echo "b 2"; echo "c 3"; echo "#0 4"; echo "#1 5") > phones.txt
  ( echo "0 1 1 1"; echo "1 2 2 2"; echo " 2 3 4 4"; echo "3 4 3 3"; echo "4 5 5 5"; echo "5 0" ) | fstcompile > in.fst
  fstcomposecontext --read-disambig-syms=disambig.list ilabels.sym in.fst tmp.fst
  fstmakecontextsyms phones.txt ilabels.sym > context.txt
  cp phones.txt phones_disambig.txt;  ( echo "#0 4"; echo "#1 5" ) >> phones_disambig.txt
  fstprint --isymbols=context.txt --osymbols=phones_disambig.txt tmp.fst

0	1	#-1	a
1	2	<eps>/a/b	b
2	3	#0	#0
3	4	a/b/c	c
4	5	#1	#1
5	6	b/c/<eps>	<eps>

*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;
    /*
        # fstcomposecontext composes efficiently with a context fst
        # that it generates.  Without --disambig-syms specified, it
        # assumes that all input symbols of in.fst are phones.
        # It adds the subsequential symbol itself (it does not
        # appear in the output so doesn't need to be specified by the user).
        # the disambig.list is a list of disambiguation symbols on the LHS
        # of in.fst.  The symbols on the LHS of out.fst are indexes into
        # the ilabels.list file, which is a kaldi-format file containing a
        # vector<vector<int32> >, which specifies what the labels mean in
        # terms of windows of symbols.
        fstcomposecontext  ilabels.sym  [ in.fst [ out.fst ] ]
         --disambig-syms=disambig.list
         --context-size=3
         --central-position=1
         --binary=false
    */

    const char *usage =
        "Composes on the left with a dynamically created context FST\n"
        "\n"
        "Usage:  fstcomposecontext <ilabels-output-file>  [<in.fst> [<out.fst>] ]\n"
        "E.g:  fstcomposecontext ilabels.sym < LG.fst > CLG.fst\n";


    ParseOptions po(usage);
    bool binary = true;
    std::string disambig_rxfilename,
        disambig_wxfilename;
    int32 context_width = 3, central_position = 1;
    int32 nonterm_phones_offset = -1;
    po.Register("binary", &binary,
                "If true, output ilabels-output-file in binary format");
    po.Register("read-disambig-syms", &disambig_rxfilename,
                "List of disambiguation symbols on input of in.fst");
    po.Register("write-disambig-syms", &disambig_wxfilename,
                "List of disambiguation symbols on input of out.fst");
    po.Register("context-size", &context_width, "Size of phone context window");
    po.Register("central-position", &central_position,
                "Designated central position in context window");
    po.Register("nonterm-phones-offset",  &nonterm_phones_offset,
                "The integer id of #nonterm_bos in your phones.txt, if present "
                "(only relevant for grammar-FST construction, see "
                "doc/grammar.dox");

    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ilabels_out_filename = po.GetArg(1),
        fst_in_filename = po.GetOptArg(2),
        fst_out_filename = po.GetOptArg(3);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

    if ( (disambig_wxfilename != "") && (disambig_rxfilename == "") )
      KALDI_ERR << "fstcomposecontext: cannot specify --write-disambig-syms if "
          "not specifying --read-disambig-syms\n";

    std::vector<int32> disambig_in;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_in))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << PrintableRxfilename(disambig_rxfilename);

    if (disambig_in.empty()) {
      KALDI_WARN << "Disambiguation symbols list is empty; this likely "
                 << "indicates an error in data preparation.";
    }

    std::vector<std::vector<int32> > ilabels;
    VectorFst<StdArc> composed_fst;

    // Work gets done here (see context-fst.h)
    if (nonterm_phones_offset < 0) {
      // The normal case.
      ComposeContext(disambig_in, context_width, central_position,
                     fst, &composed_fst, &ilabels);
    } else {
      // The grammar-FST case. See ../doc/grammar.dox for an intro.
      if (context_width != 2 || central_position != 1) {
        KALDI_ERR << "Grammar-fst graph creation only supports models with left-"
            "biphone context.  (--nonterm-phones-offset option was supplied).";
      }
      ComposeContextLeftBiphone(nonterm_phones_offset,  disambig_in,
                                *fst, &composed_fst, &ilabels);
    }
    WriteILabelInfo(Output(ilabels_out_filename, binary).Stream(),
                    binary, ilabels);

    if (disambig_wxfilename != "") {
      std::vector<int32> disambig_out;
      for (size_t i = 0; i < ilabels.size(); i++)
        if (ilabels[i].size() == 1 && ilabels[i][0] <= 0)
          disambig_out.push_back(static_cast<int32>(i));
      if (!WriteIntegerVectorSimple(disambig_wxfilename, disambig_out)) {
        std::cerr << "fstcomposecontext: Could not write disambiguation symbols to "
                  << PrintableWxfilename(disambig_wxfilename) << '\n';
        return 1;
      }
    }

    WriteFstKaldi(composed_fst, fst_out_filename);
    delete fst;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
