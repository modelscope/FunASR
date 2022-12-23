#!/bin/bash
if [[ $OSTYPE == 'darwin'* ]]; then
  conda install -c conda-forge -y pynini=2.1.5
else
  pip install pynini==2.1.5
fi