// fstbin/fstphicompose.cc

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
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

/*
The following commands represent a basic test of this program.

cat <<EOF | fstcompile > a.fst
0  1  10 10
0  1  11 11
1
EOF

cat <<EOF | fstcompile > g.fst
0  1  10 10  2.0
0  2  100 100 6.6
2  1  10 10  0.0
2  1  11  11 1.0
1
EOF
fstcompose a.fst g.fst | fstprint
# gives, as expected:
# 0    1    10    10    2
# 1
fstphicompose 100 a.fst g.fst | fstprint
# gives, again correctly:
#0    1    10    10    2
#0    1    11    11    7.5999999
#1
 
Next, test that it's working as desired for final-probs,
i.e. it takes the backoff arc when looking for a final-prob,
only if no final-prob present at current state.

cat <<EOF | fstcompile > a.fst
0  1  10 10
0  1  11 11
1
EOF
cat <<EOF | fstcompile > g.fst
0  1  10 10  2.0
0  3  11 11  2.0
1  2  100 110 6.6
3  10.0
3  2  100 110 0.0
2
EOF
fstphicompose 100 a.fst g.fst | fstprint
# output is:
#0    1    10    10    2
#0    2    11    11    2
#1    6.5999999
#2    10

*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;
    /*
      fstphicompose does composition, but treats the second FST
      specially (basically, like a backoff LM); whenever the
      composition algorithm fails to find a match for a label
      in the second FST, if it sees a phi transition it will
      take it instead, and look for a match at the destination.
      phi is the label on the input side of the backoff arc of
      the LM (the label on the output side doesn't matter).

      Also modifies the second fst so that it treats final-probs
      "correctly", i.e. takes the failure transition when looking
      for a final-prob.  This would not work if there were
      epsilons.
    */
      
    const char *usage =
        "Composition, where the right FST has \"failure\" (phi) transitions\n"
        "that are only taken where there was no match of a \"real\" label\n"
        "You supply the label corresponding to phi.\n"
        "\n"
        "Usage:  fstphicompose phi-label (fst1-rxfilename|fst1-rspecifier) "
        "(fst2-rxfilename|fst2-rspecifier) [(out-rxfilename|out-rspecifier)]\n"
        "E.g.: fstphicompose 54 a.fst b.fst c.fst\n"
        "or: fstphicompose 11 ark:a.fsts G.fst ark:b.fsts\n";
        
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        phi_str = po.GetArg(1),
        fst1_in_str = po.GetArg(2),
        fst2_in_str = po.GetArg(3),
        fst_out_str = po.GetOptArg(4);
    
    bool is_table_1 =
        (ClassifyRspecifier(fst1_in_str, NULL, NULL) != kNoRspecifier),
        is_table_2 =
        (ClassifyRspecifier(fst2_in_str, NULL, NULL) != kNoRspecifier),
        is_table_out =
        (ClassifyWspecifier(fst_out_str, NULL, NULL, NULL) != kNoWspecifier);

    int32 phi_label;
    if (!ConvertStringToInteger(phi_str, &phi_label)
        || phi_label <= 0)
      KALDI_ERR << "Invalid first argument (phi label), expect positive integer.";
    
    if (is_table_out != (is_table_1 || is_table_2))
      KALDI_ERR << "Incompatible combination of archives and files";
    
    if (!is_table_1 && !is_table_2) { // Only dealing with files...
      VectorFst<StdArc> *fst1 = ReadFstKaldi(fst1_in_str);
      
      VectorFst<StdArc> *fst2 = ReadFstKaldi(fst2_in_str);

      PropagateFinal(phi_label, fst2); // makes it work correctly
      // w.r.t. final-probs.
      
      VectorFst<StdArc> composed_fst;

      PhiCompose(*fst1, *fst2, phi_label, &composed_fst);
      
      delete fst1;
      delete fst2;

      WriteFstKaldi(composed_fst, fst_out_str);
      return 0;
    } else if (is_table_1 && !is_table_2) {

      VectorFst<StdArc> *fst2 = ReadFstKaldi(fst2_in_str);
      PropagateFinal(phi_label, fst2); // makes it work correctly
      // w.r.t. final-probs.
      SequentialTableReader<VectorFstHolder> fst1_reader(fst1_in_str);
      TableWriter<VectorFstHolder> fst_writer(fst_out_str);
      int32 n_done = 0;
      for (; !fst1_reader.Done(); fst1_reader.Next(), n_done++) {
        VectorFst<StdArc> fst1(fst1_reader.Value());
        VectorFst<StdArc> fst_out;
        PhiCompose(fst1, *fst2, phi_label, &fst_out);
        fst_writer.Write(fst1_reader.Key(), fst_out);
      }
      KALDI_LOG << "Composed " << n_done << " FSTs.";
      return (n_done != 0 ? 0 : 1);
    } else {
      KALDI_ERR << "The combination of tables/non-tables that you "
                << "supplied is not currently supported.  Either implement this, "
                << "ask the maintainers to implement it, or call this program "
                << "differently.";
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
