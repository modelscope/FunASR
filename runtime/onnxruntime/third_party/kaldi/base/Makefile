# Whenever make is run in this directory, call ./get_version.sh as the
# first thing. This script regenereates ./version.h if necessary, e.g.
# if it does not already exist or if the version number has changed.
LOG := $(shell ./get_version.sh; echo " $$?")
ifneq ($(strip $(LOG)), 0)
  RC := $(lastword $(LOG))
  OUT := $(wordlist 1,$(shell echo $$(($(words $(LOG))-1))),$(LOG))
  ifeq ($(RC),0)
    $(info $(OUT))
  else
    $(error $(OUT))
  endif
endif

all:

include ../kaldi.mk

TESTFILES = kaldi-math-test io-funcs-test kaldi-error-test timer-test

OBJFILES = kaldi-math.o kaldi-error.o io-funcs.o kaldi-utils.o timer.o

LIBNAME = kaldi-base

ADDLIBS =

include ../makefiles/default_rules.mk
