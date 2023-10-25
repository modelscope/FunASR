
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

TESTFILES = determinize-star-test \
      pre-determinize-test trivial-factor-weight-test \
      context-fst-test factor-test table-matcher-test fstext-utils-test \
      remove-eps-local-test lattice-weight-test  \
      determinize-lattice-test lattice-utils-test deterministic-fst-test \
      push-special-test epsilon-property-test prune-special-test

OBJFILES = push-special.o kaldi-fst-io.o context-fst.o grammar-context-fst.o


LIBNAME = kaldi-fstext

# tree and matrix archives needed for test-context-fst
# matrix archive needed for push-special.
ADDLIBS = ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
