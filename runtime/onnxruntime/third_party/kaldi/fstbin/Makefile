
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

BINFILES = fstdeterminizestar  \
           fstrmsymbols fstisstochastic fstminimizeencoded fstmakecontextfst \
           fstmakecontextsyms fstaddsubsequentialloop fstaddselfloops  \
           fstrmepslocal fstcomposecontext fsttablecompose fstrand \
           fstdeterminizelog fstphicompose fstcopy \
           fstpushspecial fsts-to-transcripts fsts-project fsts-union \
           fsts-concat make-grammar-fst

OBJFILES =

TESTFILES =

# actually, this library is currently empty.  Everything is a header.
LIBFILE =

ADDLIBS = ../decoder/kaldi-decoder.a ../fstext/kaldi-fstext.a \
          ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
