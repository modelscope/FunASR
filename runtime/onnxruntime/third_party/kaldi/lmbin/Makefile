
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = arpa2fst arpa-to-const-arpa

OBJFILES =

TESTFILES =

ADDLIBS = ../lm/kaldi-lm.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
