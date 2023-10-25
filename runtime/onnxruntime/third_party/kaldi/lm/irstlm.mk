# Additionnal definitions needed to make with IRSTLM toolkit

# Assumes IRSTLM includes and libraries have been installed under 
#   $(SRCDIR)/../lmtoolkit/include/irstlm and $(SRCDIR)/../lmtoolkit/lib/irstlm

EXTRA_CXXFLAGS = -DHAVE_IRSTLM -I$(SRCDIR)/../lmtoolkit/include -Wno-sign-compare
EXTRA_LDLIBS = $(SRCDIR)/../lmtoolkit/lib/irstlm/x86_64-apple-darwin10.0/libirstlm.a -lz
